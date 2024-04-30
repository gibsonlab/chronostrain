import argparse
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--db_json_path', required=True, type=str)
    parser.add_argument('-i', '--index_path', required=True, type=str)
    parser.add_argument('--esch_shig_only', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    accessions = []
    with open(args.db_json_path, 'r') as json_file:
        entries = json.load(json_file)
    for entry in entries:
        if args.esch_shig_only:
            if entry['genus'] == 'Escherichia' or entry['genus'] == 'Shigella':
                accessions.append(entry['id'])
        else:
            accessions.append(entry['id'])

    df = pd.read_csv(args.index_path, sep='\t')
    for accession in accessions:
        seq_path = df.loc[df['Accession'] == accession, 'SeqPath'].item()
        print(seq_path)


if __name__ == "__main__":
    main()
