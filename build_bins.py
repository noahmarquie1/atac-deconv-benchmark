import snapatac2 as snap
from snapatac2.genome import hg38
from pathlib import Path
from mosaic.reference import create_barcode_mapping
import pandas as pd

#print("Importing fragments...")
#data = snap.pp.import_fragments(
#    Path("sample01_data/fragments/fragments.tsv"),
#    chrom_sizes=snap.genome.hg38,
#    sorted_by_barcode=False,
#    file=Path("binned_fragments/adata.h5ad"),
#)
print("Finished importing fragments. Adding tile matrix...")
data = snap.read("binned_fragments/adata.h5ad", backed="r+")

barcode_mapping = create_barcode_mapping("cluster_labels.txt")
barcode_mapping = barcode_mapping[~barcode_mapping.index.duplicated(keep='first')]

print("Finished creating barcode mapping.")
#snap.pp.add_tile_matrix(
#    data,
#    bin_size=500,
#    inplace=True
#)
#print(data)

data.obs["cell_type"] = barcode_mapping.reindex(data.obs_names).fillna("Unknown")
cell_type_data = snap.tl.aggregate_X(
    data,
    groupby="cell_type"
)

counts = cell_type_data.X[:]
count_matrix_df = pd.DataFrame(
    counts,
    index=cell_type_data.obs_names,
    columns=cell_type_data.var_names
)
print(count_matrix_df.head(10))

print(data.X.shape)
print(data.X)

