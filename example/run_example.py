from mosaic.deconvolve import nnls_deconvolve, elastic_net_deconvolve
from mosaic.evaluate import evaluate_deconvolution, get_true_proportions
from mosaic.reference import create_barcode_mapping
from mosaic.signature import *

TEST_PEAKS = 100_000
TEST_FRAGMENTS = 500_000
TEST = True

def run_example():
    universe = build_peak_universe([
        "peaks/sample01_filtered.narrowPeak",
        "peaks/sample02_filtered.narrowPeak"
    ])
    universe = universe.head(TEST_PEAKS)

    barcode_mapping = create_barcode_mapping("cluster_labels.txt")
    cell_types = barcode_mapping.unique()

    sorted_fragments = {ct: f"sorted_fragments/{ct}_fragments.tsv" for ct in cell_types}
    count_matrix = build_count_matrix(
        sorted_fragments,
        universe,
        max_fragments=TEST_FRAGMENTS if TEST else None
    )
    normalised = quantile_normalize(count_matrix)
    filtered = filter_below_median(normalised)

    cell_type_map = pd.Series({ct: ct for ct in filtered.columns})
    signature_matrix = build_signature_matrix(filtered, cell_type_map)
    print("Signature Matrix Generated:")
    print(signature_matrix.head())

    mixture_vector = build_mixture_vector(
        'sample03_data/fragments/SRR13252436_fragments.tsv',
        universe,
        signature_matrix,
        max_fragments=TEST_FRAGMENTS
    )
    print("Mixture Vector Generated:")
    print(mixture_vector[:5])

    #print("Shapes of matrices:")
    #print(f"  Signature matrix: {signature_matrix.shape}")
    #print(f"  Mixture vector: {mixture_vector.shape}")

    nnls_est_prop = nnls_deconvolve(signature_matrix, mixture_vector)
    elastic_est_prop = elastic_net_deconvolve(signature_matrix, mixture_vector)
    true_proportions = get_true_proportions(
        "sample03_data/fragments/SRR13252436_fragments.tsv",
        barcode_mapping,
        max_fragments=TEST_FRAGMENTS
    )
    true_proportions = true_proportions.reindex(nnls_est_prop.index, fill_value=0.0)
    evaluate_deconvolution(nnls_est_prop, true_proportions)
    evaluate_deconvolution(elastic_est_prop, true_proportions)