import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clusters_file', type=str, required=True,
                        help='<Required> The clusters file output of strainest snvclust.')
    return parser.parse_args()


def main():
    args = parse_args()

    representative = set()
    with open(args.clusters_file, 'rt') as f:
        for line in f:
            line = line.strip()
            tokens = line.split(' ')
            strain_id = tokens[1]
            representative.add(strain_id)
    for strain_id in representative:
        print(strain_id)


if __name__ == "__main__":
    main()
