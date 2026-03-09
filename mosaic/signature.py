import pandas as pd
import numpy as np
from scipy.stats import rankdata
import snapatac2 as snap
from snapatac2.genome import hg38
from pathlib import Path
import os
import shutil

def build_peak_index(universe: pd.DataFrame) -> dict:
    index = {}
    for i, row in universe.iterrows():
        index.setdefault(row["chrom"], []).append(
            (int(row["start"]), int(row["end"]), i)
        )
    for chrom in index:
        index[chrom].sort(key=lambda x: x[0])
    return index


def build_peak_universe(narrowpeak_files: list[str]) -> pd.DataFrame:
    all_peaks: list[pd.DataFrame] = []
    for f in narrowpeak_files:
        df = pd.read_csv(
            f, sep="\t", header=None,
            names=["chrom", "start", "end"],
            usecols=[0, 1, 2]
        )
        all_peaks.append(df)

    combined: pd.DataFrame = pd.concat(all_peaks, ignore_index=True)
    combined = combined.sort_values(["chrom", "start"]).reset_index(drop=True)

    merged = []
    cur_chrom, cur_start, cur_end = None, None, None

    for row in combined.itertuples(index=False):
        if cur_chrom is None:
            cur_chrom, cur_start, cur_end = row.chrom, row.start, row.end
        elif row.chrom != cur_chrom or row.start >= cur_end:
            merged.append((cur_chrom, cur_start, cur_end))
            cur_chrom, cur_start, cur_end = row.chrom, row.start, row.end
        else:
            cur_end = max(cur_end, row.end)

    if cur_chrom is not None:
        merged.append((cur_chrom, cur_start, cur_end))

    universe = pd.DataFrame(merged, columns=["chrom", "start", "end"])
    universe["peak_id"] = (
            universe["chrom"] + ":" +
            universe["start"].astype(str) + "-" +
            universe["end"].astype(str)
    )
    return universe


def count_fragments(fragments_file: str, universe: pd.DataFrame,
                    max_fragments: int = None) -> pd.Series:
    peak_index = build_peak_index(universe)
    counts = np.zeros(len(universe), dtype=np.int64)
    n_fragments = 0

    with open(fragments_file, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            chrom = parts[0]
            frag_start, frag_end = int(parts[1]), int(parts[2])
            weight = int(parts[4]) if len(parts) >= 5 else 1

            for pk_start, pk_end, pk_idx in peak_index.get(chrom, []):
                if pk_start >= frag_end:
                    break
                if frag_start < pk_end:
                    counts[pk_idx] += weight

            n_fragments += 1
            if max_fragments and n_fragments >= max_fragments:
                break

    return pd.Series(counts, index=universe["peak_id"])


def build_count_matrix(sample_fragments: dict[str, str],
                       universe: pd.DataFrame,
                       max_fragments: int = None) -> pd.DataFrame:
    columns = {}
    for sample_name, frag_path in sample_fragments.items():
        print(f"Counting fragments for {sample_name} ...")
        columns[sample_name] = count_fragments(frag_path, universe, max_fragments)
    return pd.DataFrame(columns, index=universe["peak_id"])


def bin_fragments(fragments_files: list[str], bin_size: int = 500):
    adatas = []
    shutil.rmtree("binned_data", ignore_errors=True)
    os.makedirs("binned_data", exist_ok=True)

    for i, path in enumerate(fragments_files):
        print(f"Counting fragments for {path} ...")
        adata = snap.pp.import_fragments(
            Path(path),
            chrom_sizes=hg38,
            sorted_by_barcode=False,
            file=Path(f"binned_data/sample_{i+1}.h5ad")
        )
        snap.pp.add_tile_matrix(adata, bin_size=bin_size)
        adatas.append((f"sample_{i+1}", adata))

    shutil.rmtree("combined_data", ignore_errors=True)
    os.makedirs("combined_data", exist_ok=True)

    combined_data = snap.AnnDataSet(adatas=adatas, filename="combined_data/combined.h5ads")
    return combined_data


def read_binned_fragments(binned_fragments_file: str) -> snap.AnnDataSet:
    return snap.read(Path(binned_fragments_file), backed="r+")


def build_binned_count_matrix(combined_data: snap.AnnDataSet) -> pd.DataFrame:
    snap.pp.select_features(
        combined_data,
        n_features=50000,
        inplace=True
    )
    selected_mask = np.array(combined_data.var['selected'])
    selected_bin_names = [name for name, sel in zip(combined_data.var_names, selected_mask) if sel]
    X = combined_data.X[:, selected_mask]
    print("Starting to build count matrix ...")

    count_matrix = pd.DataFrame(
        X.T.toarray(),  # (n_bins × n_cells)
        index=selected_bin_names,
        columns=combined_data.obs_names
    )
    return count_matrix


# Quality control
def quantile_normalize(count_matrix: pd.DataFrame) -> pd.DataFrame:
    arr = count_matrix.values.astype(float)
    target = np.sort(arr, axis=0).mean(axis=1)

    normalised = np.empty_like(arr)
    for col in range(arr.shape[1]):
        ranks = (rankdata(arr[:, col], method="average") - 1).astype(int)
        normalised[:, col] = target[ranks]

    return pd.DataFrame(normalised,
                        index=count_matrix.index,
                        columns=count_matrix.columns)


def filter_below_median(normalised_matrix: pd.DataFrame) -> pd.DataFrame:
    row_means = normalised_matrix.mean(axis=1)
    return normalised_matrix.loc[row_means >= row_means.median()]


# Building signature matrix and bulk mixture
def build_signature_matrix(filtered_matrix: pd.DataFrame,
                            cell_type_map: pd.Series,
                            n_peaks: int = 150) -> pd.DataFrame:
    cell_type_means = filtered_matrix.T.groupby(cell_type_map).mean().T

    background = cell_type_means.mean(axis=1).replace(0, 1e-6)
    fold_change = cell_type_means.div(background, axis=0)

    best_B = None
    best_cond = np.inf
    n_min = 50
    n_max = 200

    for n in range(n_min, n_max + 1, 10):
        selected_peaks = set()
        for ct in fold_change.columns:
            top = fold_change[ct].nlargest(n).index.tolist()
            selected_peaks.update(top)

        B_raw = cell_type_means.loc[list(selected_peaks)]
        B_zscore = (B_raw - B_raw.mean(axis=0)) / B_raw.std(axis=0).replace(0, 1)

        cond = np.linalg.cond(B_zscore.values)
        if cond < best_cond:
            best_cond = cond
            best_B_raw = B_raw

    print(
        f"Signature matrix: {best_B_raw.shape[0]} peaks x {best_B_raw.shape[1]} cell types | condition number: {best_cond:.2f}")
    return best_B_raw


def build_mixture_vector(mixture_fragments_file: str,
                         universe: pd.DataFrame,
                         signature_matrix: pd.DataFrame,
                         max_fragments: int = None) -> pd.Series:

    print("Counting fragments for bulk mixture ...")
    full_counts = count_fragments(mixture_fragments_file, universe, max_fragments)
    m = full_counts.reindex(signature_matrix.index)
    n_missing = m.isna().sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} peaks had no counts in mixture, filling with 0")
        m = m.fillna(0)

    m = full_counts.reindex(signature_matrix.index).fillna(0)
    total = m.sum()
    if total > 0:
        m = m / total * 1e6

    print(f"  Mixture vector length: {len(m)} peaks")
    return m


def build_binned_mixture_vector(fragments_file: str,
                                signature_matrix: pd.DataFrame,
                                bin_size: int = 500) -> pd.Series:

    adata = snap.pp.import_fragments(
        Path(fragments_file),
        chrom_sizes=hg38,
        sorted_by_barcode=False,
        file=Path("binned_fragments/bulk_mixture.h5ad"),
    )

    snap.pp.add_tile_matrix(
        adata,
        bin_size=bin_size,
        inplace=True
    )

    count_matrix = adata.X
    bin_names = adata.var_names
    bulk_counts = count_matrix[0, :].toarray().flatten()
    mixture_vector = pd.Series(bulk_counts, index=bin_names, name='bulk_mixture')
    mixture_vector = mixture_vector.reindex(signature_matrix.index, fill_value=0)
    return mixture_vector