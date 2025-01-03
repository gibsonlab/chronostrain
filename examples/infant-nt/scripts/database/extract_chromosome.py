from pathlib import Path
from Bio import SeqIO
from chronostrain.util.io import read_seq_file
import click


def extract_chromosomes(path: Path, out_path: Path):
    for record in read_seq_file(path, file_format='fasta'):
        desc = record.description
        if "plasmid" in desc or len(record.seq) < 500000:
            continue
        else:
            SeqIO.write([record], out_path, "fasta")


@click.command()
@click.option(
    '--in', '-i', 'input_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    required=True,
    help="The source FASTA file (possibly multi-fasta)."
)
@click.option(
    '--out', '-o', 'output_path',
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="The target FASTA file path."
)
def main(input_path: Path, output_path: Path):
    extract_chromosomes(input_path, output_path)


if __name__ == "__main__":
    main()

