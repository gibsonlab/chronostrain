import argparse
from typing import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clusters", required=True)
    parser.add_argument("-o", "--out_path", required=True)
    parser.add_argument("-s", "--simulation", required=True,
                        help='The path to the CSV file that contains the strains/abundances to be simulated.')
    return parser.parse_args()


def pick_representative(clust: List[str], strains_to_sim: Set[str]):
    for member in clust:
        if member not in strains_to_sim:
            return member
    raise ValueError("Couldn't pick a representative for cluster.")


def main():
    args = parse_args()

    with open(args.simulation, "rt") as sim_f:
        header = sim_f.readline()
        strains_to_sim = set(header.strip().split(",")[1:])
        print("Strains for simulation: {}".format(strains_to_sim))

    with open(args.clusters, "rt") as clust_f, open(args.out_path, "w") as out_f:
        for line_idx, line in enumerate(clust_f):
            clust = line.strip().split('\t')

            # pick the representative
            try:
                rep = pick_representative(clust, strains_to_sim)
            except ValueError as e:
                raise ValueError("Couldn't pick a representative for cluster on line {}.".format(line_idx+1)) from None

            print(rep, file=out_f)


if __name__ == "__main__":
    main()
