import argparse
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--db_json_path', required=True, type=str)
    parser.add_argument('-i', '--index_path', required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    accessions = []
    with open(args.db_json_path, 'r') as json_file:
        entries = json.load(json_file)
    for entry in entries:
        if entry['species'] == 'coli':
            accessions.append(entry['id'])

    df = pd.read_csv(args.index_path, sep='\t')
    for accession in accessions:
        seq_path = df.loc[df['Accession'] == accession, 'SeqPath'].item()
        print(seq_path)


if __name__ == "__main__":
    main()
