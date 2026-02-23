import os
import subprocess

sampleName: str
accession: str

def run_chromap(fastq_r1, fastq_r2, barcode_fastq, genome_index, reference_fa, whitelist, threads, output):
    cmd = [
        'chromap',
        '--preset', 'atac',
        '-t', str(threads),
        '-x', genome_index,
        '-r', reference_fa,
        '-1', fastq_r1,
        '-2', fastq_r2,
        '-b', barcode_fastq,
        '--barcode-whitelist', whitelist,
        '--remove-pcr-duplicates',
        '-o', output
    ]

    subprocess.run(cmd, check=True)


def download_fragments(accession: str, threads: int):
    # Confirm that all four fragment files exist
    lengths = {}
    for i in range(1, 5):
        fastq_str = accession + "_" + str(i) + ".fastq"
        path = f"./{sampleName}_data/fastq/{fastq_str}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fragment file {fastq_str} not found. Please download the full run.")
        else:
            print(f"Fragment file {fastq_str} found.")
            with open(path, "r") as f:
                head = f.readline()
            length = head.split("length=")[1].split("\n")[0]
            lengths[str(i)] = length

    print("All fragment files found. Proceeding to identify reads.")
    r1 = None
    r2 = None
    barcodes = None

    for read, length in lengths.items():
        if not r1 and length == "101":
            r1 = read
        elif length == "101":
            r2 = read
        elif length == "16":
            barcodes = read
    if not r1 or not r2 or not barcodes:
        raise ValueError("Could not identify reads. Please download the full run.")

    print(f"Identified reads: R1={r1}, R2={r2}, Barcodes={barcodes}. Proceeding with download.")

    r1 = f"./{sampleName}_data/fastq/{accession}_{r1}.fastq"
    r2 = f"./{sampleName}_data/fastq/{accession}_{r2}.fastq"
    barcodes = f"./{sampleName}_data/fastq/{accession}_{barcodes}.fastq"
    output_dir = f"./{sampleName}_data/fragments/"
    output_path = output_dir + f"{accession}_fragments.tsv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_chromap(
        fastq_r1=r1,
        fastq_r2=r2,
        barcode_fastq=barcodes,
        genome_index="./hg38.index",
        reference_fa="./hg38.fa",
        whitelist="./737K-cratac-v1_rc.txt",
        threads=threads,
        output=output_path
    )


if __name__ == '__main__':

    sampleName = input("Please enter desired sample name:\n")
    accession = input("Please enter accession number:\n")
    download_option = input("Would you like to download the fragments for this run? \n   - (0): No \n   - (1): Yes\n")

    threads = ""
    while type(threads) != int:
        try:
            threads = int(input("Enter number of threads to use for download: "))
        except ValueError:
            print("Invalid input. Please enter an integer.")

    path = f"./{sampleName}_data/"
    os.makedirs(path, exist_ok=True)

    sra_dir = path + "SRA/"
    if not os.path.exists(sra_dir):
        os.makedirs(sra_dir)
    subprocess.run(['prefetch', accession, '-O', sra_dir], check=True)
    sra_path = sra_dir + accession + "/" + accession + ".sra"
    output_dir = path + "fastq/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    subprocess.run(['fasterq-dump', '--split-files', '--include-technical', '-O', output_dir, sra_path], check=True)
    print(f"Download complete: {accession}")

    match download_option.strip():
        case "0":
            print("Exiting...")
            exit()
        case "1":
            print("Downloading fragments...")
            download_fragments(accession=accession, threads=threads)



