from pathlib import Path
import snapatac2 as snap
from snapatac2.genome import hg38

adata = snap.pp.import_fragments(
    Path("sample01_data/fragments/fragments.tsv"),
    chrom_sizes=hg38,
    file=Path("sample01_data/output/sample_01_anndata.h5ad"),
    min_num_fragments=0,
    sorted_by_barcode=False
)
print("Finished importing fragments.")

snap.pp.add_tile_matrix(adata)
snap.pp.select_features(adata, n_features=1000)
adata.close()

