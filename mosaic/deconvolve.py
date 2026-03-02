import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet

def nnls_deconvolve(signature_matrix: pd.DataFrame,
               mixture_vector: pd.Series) -> pd.Series:

    m = mixture_vector.reindex(signature_matrix.index)
    f, residual = nnls(signature_matrix.values, m.values)

    if f.sum() > 0:
        f = f / f.sum()

    proportions = pd.Series(f, index=signature_matrix.columns)

    print_proportions(proportions)
    print(f"\n  Residual: {residual:.4f}")

    return proportions


def elastic_net_deconvolve(signature_matrix: pd.DataFrame,
                           mixture_vector: pd.Series) -> pd.Series:

    model = ElasticNet(alpha=0.01, l1_ratio=0.5, positive=True)
    model.fit(signature_matrix, mixture_vector)
    proportions = pd.Series(model.coef_, index=signature_matrix.columns)

    total = proportions.sum()
    if total > 0:
        proportions = proportions / total

    print_proportions(proportions)
    return proportions


def print_proportions(proportions: pd.Series):
    print("\nEstimated cell type proportions:")
    print("─" * 35)
    for cell_type, proportion in proportions.sort_values(ascending=False).items():
        bar = "█" * int(proportion * 40)
        print(f"  {cell_type:<25} {proportion:.4f}  {bar}")
    print("─" * 35)
    print(f"  {'Total':<25} {proportions.sum():.4f}")