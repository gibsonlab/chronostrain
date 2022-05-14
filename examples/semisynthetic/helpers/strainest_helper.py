import argparse
from typing import List
from pathlib import Path
import pandas as pd

from chronostrain import cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index_path', required=True, type=str)
    parser.add_argument('-o', '--output_fasta', required=True, type=str)
    parser.add_argument('-r', '--rep_fasta', required=True, type=str)
    parser.add_argument('-t', '--target_script_path', required=True, type=str)
    return parser.parse_args()


def strainest_mapgenomes(genome_paths: List[Path], rep_fasta: Path, output_fasta: Path):
    script = 'strainest mapgenomes {} {} {}'.format(
        ' '.join(str(genome_paths)),
        rep_fasta,
        output_fasta
    )
    return script


def main():
    args = parse_args()
    db = cfg.database_cfg.get_database()

    accessions = []
    for strain in db.all_strains():
        if strain.metadata.species == 'coli':
            accessions.append(strain.id)

    genome_paths = []
    df = pd.read_csv(args.index_path, sep='\t')
    for accession in accessions:
        seq_path = df.loc[df['Accession'] == accession, 'SeqPath'].item()
        genome_paths.append(seq_path)

    script = strainest_mapgenomes(
        genome_paths,
        Path(args.rep_fasta),
        Path(args.output_fasta)
    )
    with open(args.target_script_path, 'w') as f:
        print(script, file=f)
