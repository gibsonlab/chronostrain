# compatible with python 2

import csv
import argparse
import itertools
from pathlib import Path
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
import pandas as pd


class SpeciesNotIncluded(BaseException):
    pass


def fetch_strain_id_from_straingst(strain_name, index_df):
    fasta_path = Path("/mnt/e/strainge/strainge_db") / strain_name
    gcf_id = '_'.join(fasta_path.resolve().stem.split('_')[:2])
    hit = index_df.loc[index_df['Assembly'] == gcf_id, :].head(1)
    if hit.shape[0] == 0:
        raise ValueError(
            "Couldn't find strain from StrainGST identifier `{}`.".format(strain_name)
        )

    if hit['Species'].item().lower() != 'coli':
        raise SpeciesNotIncluded()

    return hit['Accession'].item(), hit['Strain'].item()


def parse_distances(similarities_path, index_df):
    print("Parsing distances from {}".format(similarities_path))
    names = set()
    dists = dict()
    names_to_ids = dict()

    def fetch_id(strain_name):
        if strain_name not in names_to_ids:
            acc, _ = fetch_strain_id_from_straingst(strain_name, index_df)
            names_to_ids[strain_name] = acc
        return names_to_ids[strain_name]

    with open(similarities_path, "rt") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                u_name = row['kmerset1']
                v_name = row['kmerset2']
                jaccard_dist = 1 - float(row['jaccard'])
                u_id = fetch_id(u_name)
                v_id = fetch_id(v_name)

                names.add(u_id)
                names.add(v_id)

                if u_id < v_id:
                    dists[(u_id, v_id)] = jaccard_dist
                else:
                    dists[(v_id, u_id)] = jaccard_dist
            except SpeciesNotIncluded:
                continue

    print("found {} records.".format(len(names)))
    names = sorted(names)
    matrix = [
        [0.] * (u_idx + 1)
        for u_idx, _ in enumerate(names)
    ]
    for (u_idx, u), (v_idx, v) in itertools.combinations(enumerate(names), r=2):
        if u_idx > v_idx:
            matrix[u_idx][v_idx] = dists[(v, u)]
        else:
            matrix[v_idx][u_idx] = dists[(u, v)]
    return DistanceMatrix(names, matrix)


def create_tree(d):
    print("Constructing tree from distance matrix.")
    tree_constructor = DistanceTreeConstructor()
    return tree_constructor.nj(d)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sim_tsv_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('-f', '--tree_format', type=str, required=False, default='newick')
    parser.add_argument('-i', '--refseq_index', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    index_df = pd.read_csv(args.refseq_index, sep='\t')
    tree = create_tree(parse_distances(args.sim_tsv_path, index_df))
    Phylo.write([tree], str(args.output_path), args.tree_format)
    print("Created tree {}".format(args.output_path))


if __name__ == "__main__":
    main()
