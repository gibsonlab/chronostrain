import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Iterator, Tuple, Dict

import pandas as pd
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--chronostrain_json', type=str, required=True)
    parser.add_argument('-i', '--refseq_index', type=str, required=True)
    parser.add_argument('-o', '--out_path', type=str, required=True)
    parser.add_argument('-f', '--tree_format', type=str, required=False, default='newick')
    return parser.parse_args()


def retrieve_chronostrain_strains(json_path: Path, index_df: pd.DataFrame) -> Iterator[Tuple[str, Path]]:
    print(f"Parsing IDs from chronostrain database file {json_path}.")
    with open(json_path, 'r') as f:
        strain_entries = json.load(f)
        for strain_entry in strain_entries:
            strain_species = strain_entry['species']
            if strain_species != 'faecalis':
                continue

            strain_id = strain_entry['id']

            # Use this project's refseq index.
            result = index_df.loc[index_df['Accession'] == strain_id].head(1)
            seq_path = result['SeqPath'].item()
            yield strain_id, seq_path


def compute_distances(
        strain_records: Iterator[Tuple[str, Path]],
        work_dir: Path
) -> DistanceMatrix:
    work_dir.mkdir(exist_ok=True, parents=True)

    strain_sketches = {}
    for strain_id, strain_path in strain_records:
        if strain_id in strain_sketches:
            continue

        print(f"Sketching {strain_id}...")
        out_prefix = work_dir / f'{strain_id}'
        sketch_path = invoke_mash_sketch(strain_path, out_prefix)
        strain_sketches[strain_id] = sketch_path

    indices: Dict[str, int] = {}
    strain_ids = []
    for i, strain_id in enumerate(strain_sketches.keys()):
        indices[strain_id] = i
        strain_ids.append(strain_id)

    matrix = [
        [0.] * k
        for k in range(1, len(indices) + 1)
    ]

    for x, y in itertools.combinations(strain_sketches.keys(), r=2):
        d = invoke_mash_dist(strain_sketches[x], strain_sketches[y])

        x_idx = indices[x]
        y_idx = indices[y]
        if x_idx > y_idx:
            matrix[x_idx][y_idx] = d
        else:
            matrix[y_idx][x_idx] = d

    return DistanceMatrix(names=strain_ids, matrix=matrix)


def invoke_mash_sketch(fasta_path: Path, out_prefix: Path) -> Path:
    expected_out_path = out_prefix.parent / f'{out_prefix.name}.msh'
    if not expected_out_path.exists():
        subprocess.run([
            'mash', 'sketch',
            str(fasta_path),
            '-o', str(out_prefix)
        ], capture_output=False)
    return expected_out_path


def invoke_mash_dist(sketch1: Path, sketch2: Path) -> float:
    completed = subprocess.run(['mash', 'dist', str(sketch1), str(sketch2)], capture_output=True)
    completed.check_returncode()

    answer = completed.stdout.decode('utf-8')
    ref_id, query_id, mash_dist, p_val, matching_hashes = answer.split('\t')
    return float(mash_dist)


def main():
    args = parse_args()
    output_path = Path(args.out_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    temp_dir = output_path.parent / '__dist_tmp'
    temp_dir.mkdir(exist_ok=True, parents=True)

    index_df = pd.read_csv(args.refseq_index, sep='\t')

    # ========== Create tree.
    chronostrain_strains = retrieve_chronostrain_strains(
        args.chronostrain_json,
        index_df
    )
    distances = compute_distances(chronostrain_strains, temp_dir)

    print("Constructing tree from distance matrix.")
    tree = DistanceTreeConstructor().nj(distances)

    # erase internal node names. Necessary for SynerClust?
    for clade in tree.get_nonterminals():
        clade.name = ""

    # Save the tree.
    Phylo.write([tree], output_path, args.tree_format)
    print("Created tree {}".format(output_path))


if __name__ == "__main__":
    main()
