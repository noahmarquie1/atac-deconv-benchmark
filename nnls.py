import pandas as pd
import numpy as np
import gzip
from scipy.stats import rankdata, mannwhitneyu
from scipy.linalg import svd
import os

TEST = True
TEST_PEAKS = 5_000
TEST_FRAGMENTS = 10_000

# Fragment processing

def build_peak_index(universe: pd.DataFrame) -> dict:
    index = {}
    for i, row in universe.iterrows():
        index.setdefault(row["chrom"], []).append(
            (int(row["start"]), int(row["end"]), int(i))
        )
    for chrom in index:
        index[chrom].sort(key=lambda x: x[0])
    return index


def count_fragments(fragments_file: str, universe: pd.DataFrame,
                    max_fragments: int = None) -> pd.Series:
    peak_index = build_peak_index(universe)
    counts = np.zeros(len(universe), dtype=np.int64)
    _open = gzip.open if fragments_file.endswith(".gz") else open
    n_fragments = 0

    with _open(fragments_file, "rt") as fh:
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


def subset_fragments_by_celltype(experiment_fragments: dict[str, str],
                                  barcode_celltype_map: pd.Series,
                                  output_dir: str) -> dict[str, str]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cell_types = barcode_celltype_map.unique()
    output_handles = {
        ct: open(f"{output_dir}/{ct}_fragments.tsv", "w")
        for ct in cell_types
    }

    for experiment_name, fragments_file in experiment_fragments.items():
        print(f"\nProcessing {experiment_name} ...")
        n_matched = 0
        n_unmatched = 0
        with open(fragments_file, "rt") as fh:
            length = sum(1 for _ in fh.readlines())

        with open(fragments_file, "rt") as fh:
            for i in range(round(length / 100)):
                line = fh.readline()

                print(f"Percent complete: {i / (length / 100) * 100:.2f}% ")

                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue

                barcode = parts[3]
                if barcode in barcode_celltype_map.index:
                    cell_type = barcode_celltype_map[barcode]
                    if isinstance(cell_type, pd.Series):
                        cell_type = cell_type.iloc[0]
                    output_handles[cell_type].write(line)
                    n_matched += 1
                else:
                    n_unmatched += 1

        print(f"  Matched:   {n_matched:,} fragments")
        print(f"  Unmatched: {n_unmatched:,} fragments")
        print(f"Percentage Matched: {n_matched / (n_matched + n_unmatched + 1):.2%}")

    for handle in output_handles.values():
        handle.close()

    return {ct: f"{output_dir}/{ct}_fragments.tsv" for ct in cell_types}


# Peak universe building and processing

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


# Working with barcodes and cell types

def process_barcode(cell_name: str) -> str:
    # Remove prefix (everything before and including '_')
    if '_' in cell_name:
        barcode = cell_name.split('_', 1)[1]
    else:
        barcode = cell_name

    # Reverse complement
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    rev_comp = ''.join(complement.get(base, base) for base in reversed(barcode))
    return rev_comp


def create_barcode_celltype_map(cluster_labels_file: str) -> pd.Series:
    labels = pd.read_csv(cluster_labels_file, sep="\t")
    processed_barcodes = labels["cellName"].apply(process_barcode)

    return pd.Series(
        labels["cluster_name"].values,
        index=processed_barcodes
    )

# building count matrix, signature matrix, bulk mixture

def build_count_matrix(sample_fragments: dict[str, str],
                       universe: pd.DataFrame,
                       max_fragments: int = None) -> pd.DataFrame:
    columns = {}
    for sample_name, frag_path in sample_fragments.items():
        print(f"Counting fragments for {sample_name} ...")
        columns[sample_name] = count_fragments(frag_path, universe, max_fragments)
    return pd.DataFrame(columns, index=universe["peak_id"])


def build_signature_matrix(filtered_matrix: pd.DataFrame,
                            cell_type_map: pd.Series,
                            n_peaks: int = 500) -> pd.DataFrame:
    # Average replicates by cell type
    cell_type_means = filtered_matrix.T.groupby(cell_type_map).mean().T

    # Select top n_peaks per cell type by value
    selected_peaks = set()
    for ct in cell_type_means.columns:
        top = cell_type_means[ct].nlargest(n_peaks).index.tolist()
        selected_peaks.update(top)

    B = cell_type_means.loc[list(selected_peaks)]

    print(f"Signature matrix: {B.shape[0]} peaks x {B.shape[1]} cell types")
    return B


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

    print(f"  Mixture vector length: {len(m)} peaks")
    return m


def deconvolve(signature_matrix: pd.DataFrame,
               mixture_vector: pd.Series) -> pd.Series:
    """
    Performs NNLS to estimate cell type proportions from bulk mixture.
    Prints results sorted by proportion.
    """
    from scipy.optimize import nnls

    # Align mixture vector to signature matrix row order
    m = mixture_vector.reindex(signature_matrix.index)

    # Solve m = B x f with non-negativity constraint
    f, residual = nnls(signature_matrix.values, m.values)

    # Normalise to proportions summing to 1
    if f.sum() > 0:
        f = f / f.sum()

    proportions = pd.Series(f, index=signature_matrix.columns)

    print("\nEstimated cell type proportions:")
    print("─" * 35)
    for cell_type, proportion in proportions.sort_values(ascending=False).items():
        bar = "█" * int(proportion * 40)
        print(f"  {cell_type:<25} {proportion:.4f}  {bar}")
    print("─" * 35)
    print(f"  {'Total':<25} {proportions.sum():.4f}")
    print(f"\n  Residual: {residual:.4f}")

    return proportions


# Evaluation Functions

def get_true_proportions(fragments_file: str,
                          barcode_celltype_map: pd.Series,
                          max_fragments: int = None) -> pd.Series:
    cell_type_counts = {}
    n = 0
    with open(fragments_file, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                barcode = parts[3]
                if barcode in barcode_celltype_map.index:
                    cell_type = barcode_celltype_map[barcode]
                    if isinstance(cell_type, pd.Series):
                        cell_type = cell_type.iloc[0]
                    cell_type_counts[cell_type] = cell_type_counts.get(cell_type, 0) + 1

            n += 1
            if n >= max_fragments:
                break

    total = sum(cell_type_counts.values())
    if total == 0:
        print("Warning: No matched barcodes found!")
        return pd.Series(dtype=float)

    return pd.Series(cell_type_counts) / total


def evaluate_deconvolution(fragments_file: str,
                           estimated_proportions: pd.Series,
                           true_proportions: pd.Series,
                           barcode_celltype_map: pd.Series,
                           max_fragments: int = None) -> dict:
    # Compute error metrics
    errors = estimated_proportions - true_proportions
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2

    rmse = np.sqrt(squared_errors.mean())
    mae = abs_errors.mean()
    max_error = abs_errors.max()

    # Pearson correlation
    if len(true_proportions) > 1 and true_proportions.std() > 0 and estimated_proportions.std() > 0:
        correlation = np.corrcoef(true_proportions.values, estimated_proportions.values)[0, 1]
    else:
        correlation = np.nan

    # Print comparison
    print("\n" + "=" * 70)
    print("DECONVOLUTION EVALUATION")
    print("=" * 70)
    print(f"{'Cell Type':<25} {'True':<10} {'Estimated':<10} {'Error':<10}")
    print("─" * 70)

    for ct in sorted(estimated_proportions.index, key=lambda x: true_proportions[x], reverse=True):
        t = true_proportions[ct]
        e = estimated_proportions[ct]
        err = e - t
        print(f"{ct:<25} {t:>9.4f}  {e:>9.4f}  {err:>+9.4f}")

    print("─" * 70)
    print(f"\nError Metrics:")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"  Max Absolute Error:             {max_error:.4f}")
    print(f"  Pearson Correlation:            {correlation:.4f}")
    print("=" * 70)

    return {
        'true_proportions': true_proportions,
        'estimated_proportions': estimated_proportions,
        'errors': errors,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'correlation': correlation
    }


if TEST:
    universe = build_peak_universe([
        "peaks/sample01_filtered.narrowPeak",
        "peaks/sample02_filtered.narrowPeak"
    ])
    universe = universe.head(TEST_PEAKS)

    sample_fragments = {
        "sample01": "sample01_data/fragments/fragments.tsv",
        "sample02": "sample02_data/fragments/SRR13252435_fragments.tsv"
    }

    celltype_map = create_barcode_celltype_map("cluster_labels.txt")
    cell_types = celltype_map.unique()
    print(celltype_map)

    #subset_fragments_by_celltype(
    #    sample_fragments,
    #    celltype_map,
    #    "sorted_fragments"
    #)
    celltype_fragments = {ct: f"sorted_fragments/{ct}_fragments.tsv" for ct in cell_types}

    count_matrix = build_count_matrix(
        celltype_fragments,  # dict returned by subset_fragments_by_celltype
        universe,  # DataFrame returned by build_peak_universe
        max_fragments=TEST_FRAGMENTS if TEST else None
    )
    normalised = quantile_normalize(count_matrix)
    filtered = filter_below_median(normalised)

    cell_type_map = pd.Series({ct: ct for ct in filtered.columns})
    signature_matrix = build_signature_matrix(filtered, cell_type_map)
    print(signature_matrix.head())

    mixture_vector = build_mixture_vector(
        'sample03_data/fragments/SRR13252436_fragments.tsv',
        universe,
        signature_matrix,
        max_fragments=TEST_FRAGMENTS
    )
    print(mixture_vector)

    print("Shapes of matrices:")
    print(f"  Signature matrix: {signature_matrix.shape}")
    print(f"  Mixture vector: {mixture_vector.shape}")

    estimated_proportions = deconvolve(signature_matrix, mixture_vector)
    true_proportions = get_true_proportions(
        "sample03_data/fragments/SRR13252436_fragments.tsv",
        celltype_map,
        max_fragments=TEST_FRAGMENTS
    )
    true_proportions = true_proportions.reindex(estimated_proportions.index, fill_value=0.0)

    evaluate_deconvolution(
        "sample03_data/fragments/SRR13252436_fragments.tsv",
        estimated_proportions,
        true_proportions,
        celltype_map,
        max_fragments=TEST_FRAGMENTS
    )