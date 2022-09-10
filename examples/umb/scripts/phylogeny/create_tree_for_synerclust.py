import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Iterator, Tuple, Dict, List

import pandas as pd
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix


def create_tree(d):
    print("Constructing tree from distance matrix.")
    tree_constructor = DistanceTreeConstructor()
    return tree_constructor.nj(d)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--strainge_strains', type=str, required=True)
    parser.add_argument('-sdb', '--strainge_db_dir', type=str, required=True)
    parser.add_argument('-j', '--chronostrain_json', type=str, required=True)
    parser.add_argument('-i', '--refseq_index', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-f', '--tree_format', type=str, required=False, default='newick')
    return parser.parse_args()


def retrieve_straingst_strains(strain_list_path: Path, strainge_db_dir: Path, index_df: pd.DataFrame) -> Iterator[Tuple[str, Path]]:
    print(f"Parsing IDs from strainGE database strains {strain_list_path}.")
    with open(strain_list_path, 'rt') as f:
        for line in f:
            relative_path = Path(line.strip())
            if relative_path.suffix == ".hdf5":
                relative_path = relative_path.with_suffix('')

            fasta_name = relative_path.name
            if not fasta_name.startswith("Esch_coli"):
                continue

            # Resolve symlinks to retrieve GCF id
            full_fasta_path = (strainge_db_dir / fasta_name).resolve()
            assert full_fasta_path.name.startswith("GCF")
            gcf_id = 'GCF_{}'.format(full_fasta_path.name.split("_")[1])

            # Use this project's refseq index.
            result = index_df.loc[index_df['Assembly'] == gcf_id].head(1)
            strain_id = result['Accession'].item()
            seq_path = result['SeqPath'].item()
            yield strain_id, seq_path


def retrieve_chronostrain_strains(json_path: Path, index_df: pd.DataFrame) -> Iterator[Tuple[str, Path]]:
    print(f"Parsing IDs from chronostrain database file {json_path}.")
    with open(json_path, 'r') as f:
        strain_entries = json.load(f)
        for strain_entry in strain_entries:
            strain_species = strain_entry['species']
            if strain_species != 'coli':
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


def create_synerclust_input(target_path: Path, strain_ids: List[str], index_df: pd.DataFrame):
    # Search for fasta path.
    with open(target_path, "w") as out_f:
        print("//", file=out_f)
        for strain_id in strain_ids:
            res = index_df.loc[index_df['Accession'] == strain_id, :]
            if res.shape[0] > 1:
                raise RuntimeError("Ambiguous accession -> gcf mapping.")

            gcf = res.head(1)['Assembly'].item()
            seq_path = Path(res.head(1)['SeqPath'].item())
            gff_path = next(iter(seq_path.parent.glob(f'{gcf}_*genomic.chrom.gff')))

            print(f"Genome\t{strain_id}", file=out_f)
            print(f"Sequence\t{seq_path}", file=out_f)
            print(f"Annotation\t{gff_path}", file=out_f)
            print("//", file=out_f)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    temp_dir = output_dir / '__dist_tmp'
    temp_dir.mkdir(exist_ok=True, parents=True)

    index_df = pd.read_csv(args.refseq_index, sep='\t')

    # ========== Create tree.
    tree_path = output_dir / "tree.nwk"
    if not tree_path.exists():
        straingst_strains = retrieve_straingst_strains(
            args.strainge_strains,
            Path(args.strainge_db_dir),
            index_df
        )
        chronostrain_strains = retrieve_chronostrain_strains(
            args.chronostrain_json,
            index_df
        )
        distances = compute_distances(
            itertools.chain(straingst_strains, chronostrain_strains),
            temp_dir
        )

        print("Constructing tree from distance matrix.")
        tree = DistanceTreeConstructor().nj(distances)

        # erase internal node names. Necessary for SynerClust?
        for clade in tree.get_nonterminals():
            clade.name = ""

        # Save the tree.
        Phylo.write([tree], tree_path, args.tree_format)
        print("Created tree {}".format(tree_path))
        print("To run SynerClust, the user might need to manually delete the root node distance (`:0.000`).")
    else:
        print("Already found pre-computed tree {}".format(tree_path))

    # Generate synerclust input.
    straingst_strains = retrieve_straingst_strains(
        args.strainge_strains,
        Path(args.strainge_db_dir),
        index_df
    )
    chronostrain_strains = retrieve_chronostrain_strains(
        args.chronostrain_json,
        index_df
    )

    strain_ids = [
        strain_id
        for strain_id, _ in itertools.chain(straingst_strains, chronostrain_strains)
    ]

    synerclust_input = output_dir / "synerclust_input.txt"
    create_synerclust_input(synerclust_input, strain_ids, index_df)
    print("Created synerclust input file {}".format(synerclust_input))


if __name__ == "__main__":
    main()
