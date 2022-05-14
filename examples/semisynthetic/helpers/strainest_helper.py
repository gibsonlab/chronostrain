import argparse
from pathlib import Path
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--db_json_path', required=True, type=str)
    parser.add_argument('-i', '--index_path', required=True, type=str)
    parser.add_argument('-o', '--output_fasta', required=True, type=str)
    parser.add_argument('-r', '--rep_fasta', required=True, type=str)
    parser.add_argument('-t', '--target_script_path', required=True, type=str)
    return parser.parse_args()


def strainest_mapgenomes(genome_paths, rep_fasta, output_fasta):
    script = 'strainest mapgenomes {} {} {}'.format(
        ' '.join(str(genome_paths)),
        rep_fasta,
        output_fasta
    )
    return script


def main():
    args = parse_args()

    accessions = []
    with open(args.db_json_path, 'r') as json_file:
        entries = json.load(json_file)
    for entry in entries:
        if entry['species'] == 'coli':
            accessions.append(entry['id'])

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


if __name__ == "__main__":
    main()
