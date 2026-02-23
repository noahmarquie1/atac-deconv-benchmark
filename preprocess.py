from pathlib import Path
import snapatac2 as snap
from snapatac2.genome import hg38
import subprocess

def fragments_to_bedpe(input_file, output_file):
    command = [
        "macs3", "filterdup",
        "-i", input_file,
        "-f", "BED",
        "--keep-dup", "all",
        "-o", output_file
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully converted {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


def call_peaks(input_file):
    command = [
        "macs3", "callpeak",
        "-t", input_file,
        "-f", "BEDPE",  # Uses the full fragment span
        "-g", "hs",  # Effective genome size
        "-n", "sample01_data/bed/macs3_out/sample01",  # Output prefix
        "-B",  # Generate bedGraph signal tracks
        "-q", "0.01",  # FDR threshold (stringent)
        "--nomodel",  # Skip model building (already have fragments)
        "--call-summits",  # Required for finding exact centers
        "--keep-dup", "all"  # Keep all fragments (standard for ATAC)
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Successfully called peaks for {input_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during peak calling: {e}")

#fragments_to_bedpe("sample01_data/fragments/fragments.tsv", "sample01_data/bed/sample01.bedpe")
call_peaks("sample01_data/bed/sample01.bedpe")


quit()
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

