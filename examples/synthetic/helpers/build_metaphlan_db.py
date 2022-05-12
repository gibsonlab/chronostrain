import os
import pickle
import bz2
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Genome(object):
    tax_with_clades: str
    taxid: str
    length: int


@dataclass
class Marker(object):
    name: str
    fasta_path: Path
    clade: str
    length: int
    taxon: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', required=True,
                        help='<Required> The directory to output the target database into.')
    return parser.parse_args()


def create_db(genomes: List[Genome], markers: List[Marker], db_dir: Path):
    db_dir.mkdir(exist_ok=True, parents=True)
    db_fasta_path = db_dir / "database.fasta"

    with open(db_fasta_path, 'w') as fasta_file:
        for marker in markers:
            with open(marker.fasta_path, 'r') as marker_file:
                for line in marker_file:
                    if line.startswith('>'):
                        print(
                            f'>{marker.name} {marker.name};{marker.taxon};CP009273',
                            file=fasta_file
                        )
                    else:
                        fasta_file.write(line)

    print(f'bowtie2-build {db_fasta_path} {db_dir}/database')
    os.system(f'bowtie2-build {db_fasta_path} {db_dir}/database')

    db = {
        'taxonomy': {
            genome.tax_with_clades: (genome.taxid, genome.length)
            for genome in genomes
        },
        'markers': {
            marker.name: {
                'clade': marker.clade,
                'ext': {},
                'len': marker.length,
                'taxon': marker.taxon
            }
            for marker in markers
        }
    }
    with bz2.BZ2File(db_dir / 'database.pkl', 'w') as db_file:
        pickle.dump(db, db_file, pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()

    db_dir = Path(args.output_dir)
    genomes = [
        Genome(
            'k__Bacteria|p_Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Escherichia|s__Escherichia_coli|t__CP009273',
            '2|1224|1236|91347|543|561|562',
            549
        )
    ]

    markers = [
        Marker(
            'fimA',
            Path('/mnt/d/chronostrain/synthetic/database/markers/CP009273.1_Original/fimA.fasta'),
            's__Escherichia_coli',
            4631469,
            'k__Bacteria|p_Proteobacteria|c__Gammaproteobacteria|o__Enterobacterales|f__Enterobacteriaceae|g__Escherichia|s__Escherichia_coli'
        )
    ]

    create_db(
        genomes=genomes,
        markers=markers,
        db_dir=db_dir
    )


if __name__ == "__main__":
    main()
