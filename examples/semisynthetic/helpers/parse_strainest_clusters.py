import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clusters_file', type=str, required=True,
                        help='<Required> The clusters file output of strainest snvclust.')
    parser.add_argument('--refseq_index', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    index_df = pd.read_csv(args.refseq_index, sep='\t')

    representative = set()
    with open(args.clusters_file, 'rt') as f:
        for line in f:
            line = line.strip()
            tokens = line.split('\t')
            strain_id = tokens[1].removesuffix('.chrom.fna')
            representative.add(strain_id)
    for strain_id in representative:
        print(index_df.loc[index_df['Accession'] == strain_id, 'SeqPath'].item())


if __name__ == "__main__":
    main()
