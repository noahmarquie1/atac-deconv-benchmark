import pandas as pd
import numpy as np
from scipy.stats import rankdata
import snapatac2 as snap
from snapatac2.genome import hg38
from pathlib import Path

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


def build_binned_count_matrix(sample_fragments: dict[str, str],
                              barcode_mapping: pd.Series,
                              bin_size: int = 500,
                              max_fragments: int = None) -> pd.DataFrame:
    adatas = []
    for i, path in enumerate(sample_fragments.values()):
        print(f"Counting fragments for {path} ...")
        adata = snap.pp.import_fragments(
            Path("sample02_data/fragments/SRR13252435_fragments.tsv"),
            chrom_sizes=hg38,
            sorted_by_barcode=False,
            file=Path(f"binned_fragments/sample{i}_adata.h5ad"),
        )
        adatas.append((f"sample{i}_adata", adata))

    combined_data = snap.AnnDataSet(adatas=adatas, filename="binned_fragments/combined.h5ads")
    barcode_mapping = barcode_mapping[~barcode_mapping.index.duplicated(keep='first')]

    combined_data.obs["cell_type"] = barcode_mapping.reindex(combined_data.obs_names).fillna("Unknown")
    cell_type_data = snap.tl.aggregate_X(
        combined_data,
        groupby="cell_type"
    )

    counts = combined_data.X[:]
    sparsity = counts.nnz / (counts.shape[0] * counts.shape[1])
    print(f"{sparsity * 100:.2f}% non-zero entries")

    count_matrix_df = pd.DataFrame(
        counts.T,
        index=cell_type_data.var_names,
        columns=cell_type_data.obs_names,
    )
    return count_matrix_df


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