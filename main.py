import snapatac2 as snap
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import scanpy as sc
import scipy

# Methods
def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

# Main Program
adata = snap.read(Path("sample01_data/output/sample_01_anndata.h5ad"), "r")

# Process Labels
labels_df = pd.read_csv(
    "sample01_data/cluster_labels.txt",
    sep='\t',
)

labels_df['cellName'] = labels_df['cellName'].str.split('_').str[1]
labels_df['cellName'] = labels_df['cellName'].apply(reverse_complement)
labels_df = labels_df[~labels_df['cellName'].duplicated(keep='first')]
labels_df.set_index('cellName', inplace=True)
labeled_cells = labels_df.index[labels_df.index.isin(adata.obs_names)]
print("Finished processing cell labels.")

# Filter AnnData Object
adata = adata.to_memory()
adata.obs['cluster_label'] = labels_df.reindex(adata.obs_names)['cluster_name'].values
selected_mask = adata.var['selected'].to_numpy()
adata_subset = adata[:, selected_mask]
adata_subset = adata_subset[~adata_subset.obs['cluster_label'].isna()].copy()

# Filter out rare cell types
min_cells = 10
cluster_counts = adata_subset.obs['cluster_label'].value_counts()
valid_clusters = cluster_counts[cluster_counts >= min_cells].index
adata_subset = adata_subset[adata_subset.obs['cluster_label'].isin(valid_clusters)].copy()

# Split into train and test sets, make test a pseudobulk
train_cells, test_cells = train_test_split(
    adata_subset.obs_names,
    test_size=0.2,
    stratify=adata_subset.obs['cluster_label'].values,  # Use .values
    random_state=42
)

train_adata = adata_subset[train_cells].copy()
test_adata = adata_subset[test_cells].copy()

train_pb = snap.tl.aggregate_X(train_adata, groupby='cluster_label')
test_pb = snap.tl.aggregate_X(test_adata, groupby='cluster_label')

# Create signature matrix and bulk mixture
signature_matrix = pd.DataFrame(
    train_pb.X,
    index=train_pb.obs_names,
    columns=train_pb.var_names
).T
cluster_sums = signature_matrix.sum(axis=0)
cluster_sums[cluster_sums == 0] = 1
signature_matrix_normalized = signature_matrix / cluster_sums

bulk_mixture = np.array(test_adata.X.sum(axis=0)).flatten()
bulk_mixture_normalized = bulk_mixture / bulk_mixture.sum()

# Represent true proportions
true_proportions = test_adata.obs['cluster_label'].value_counts(normalize=True).to_dict()
cluster_order = signature_matrix.columns
true_proportions = np.array([
    true_proportions.get(cluster, 0.0) for cluster in cluster_order
])

# Perform NNLS and Evaluate Results
predicted_proportions, residue = nnls(
    signature_matrix_normalized.values,
    bulk_mixture_normalized
)
predicted_proportions = predicted_proportions / np.sum(predicted_proportions)

results = dict(zip(signature_matrix.columns, predicted_proportions))

rmse = np.sqrt(mean_squared_error(true_proportions, predicted_proportions))
r2 = r2_score(true_proportions, predicted_proportions)
pearson_corr, p_value = pearsonr(true_proportions, predicted_proportions)

print(f"\nEvaluation Metrics:")
print("==================")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")

print("==================")
print("\nFinished.")

