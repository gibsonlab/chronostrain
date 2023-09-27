from typing import *
from pathlib import Path
import click

import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm


def prepare_mash_input(index_df: pd.DataFrame, work_dir: Path) -> Path:
    work_dir.mkdir(exist_ok=True, parents=True)

    mash_input = work_dir / 'mash_input.txt'
    with open(mash_input, 'w') as f:
        for _, row in index_df.iterrows():
            seq_path = row['SeqPath']
            print(seq_path, file=f)
    return mash_input


def load_mash_dist(dist_tsv: Path, index_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    suffix = '.chrom.fna'
    n = index_df.shape[0]
    dist_matrix = np.zeros((n, n), dtype=float)

    accs = [row['Accession'] for _, row in index_df.iterrows()]
    acc_to_idx = {acc: i for i, acc in enumerate(accs)}

    with open(dist_tsv, 'rt') as f:
        for line in tqdm(f, total=int(n*n)):
            tokens = line.strip().split('\t')
            x = Path(tokens[0]).name[:-len(suffix)]
            y = Path(tokens[1]).name[:-len(suffix)]

            idx1 = acc_to_idx[x]
            idx2 = acc_to_idx[y]
            if idx1 == idx2:
                continue
            else:  # only fill in lower triangular entries.
                dist = float(tokens[2])
                dist_matrix[idx1, idx2] = dist
    return dist_matrix, accs


def compute_distances(
        index_df: pd.DataFrame,
        work_dir: Path,
        n_threads: int
) -> Tuple[np.ndarray, List[str]]:
    work_dir.mkdir(exist_ok=True, parents=True)

    # Prepare mash input
    mash_input_path = prepare_mash_input(index_df, work_dir)

    # Create compound sketch of all ref genomes.
    mash_output_path = work_dir / 'reference.msh'
    if not mash_output_path.exists():
        print("Invoking mash sketch.")
        subprocess.call(
            [
                'mash', 'sketch',
                '-l', str(mash_input_path),
                '-o', str(mash_output_path.with_suffix(''))
            ]
        )
    else:
        print("Mash sketch result already found.")

    # Using compound sketch, calculate all-to-all distances.
    distance_path = work_dir / 'distances.tab'
    if not distance_path.exists():
        print("Invoking mash dist on compound sketch.")
        with open(distance_path, 'w') as out_f:
            subprocess.run(
                [
                    'mash', 'dist', '-p', str(n_threads), str(mash_output_path), str(mash_output_path)
                ],
                stdout=out_f,
            )
    else:
        print("Mash dist result already found.")

    # Parse distances file.
    return load_mash_dist(distance_path, index_df)


def path_to_seq(acc: str, index_df: pd.DataFrame) -> Path:
    res = index_df.loc[index_df['Accession'] == acc, :]
    if res.shape[0] == 0:
        raise ValueError("No hits found for acc = {}".format(acc))
    return Path(res['SeqPath'].item())


def fix_accession(broken_acc):
    # this function is necessary because poppunk converts periods (.) into underscores.
    tokens = broken_acc.split("_")
    prefix = "_".join(tokens[:-1])
    suffix = tokens[-1]
    return f'{prefix}.{suffix}'


@click.command()
@click.option(
    '--index-path', '-i', 'index_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True
)
@click.option(
    '--work-dir', '-w', 'work_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True
)
@click.option(
    '--poppunk-clust', '-p', 'poppunk_clust_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True
)
@click.option(
    '--out', '-o', 'out_path',
    type=click.Path(path_type=Path, dir_okay=False),
    required=True
)
@click.option(
    '--threads', '-t', 'n_threads',
    type=int, default=1
)
def main(index_path: Path, poppunk_clust_path: Path, work_dir: Path, out_path: Path, n_threads: int):
    index_df = pd.read_csv(index_path, sep='\t')
    index_df = index_df.loc[index_df['Species'] == 'coli']
    distances, acc_ordering = compute_distances(index_df, work_dir, n_threads=n_threads)

    print("Loading poppunk clustering from {}.".format(poppunk_clust_path.name))
    poppunk_df = pd.read_csv(poppunk_clust_path, sep=',')
    acc_idxs = {
        s: i for i, s in enumerate(acc_ordering)
    }

    print("Generating output file: {}".format(out_path.name))
    with open(out_path, 'w') as out_file:
        for clust_id, section in poppunk_df.groupby("Cluster"):
            clust_members = [fix_accession(taxon_id) for taxon_id in section['Taxon']]
            indices = [acc_idxs[s] for s in clust_members if s in acc_idxs]
            if len(indices) == 0:
                continue
            submat = distances[
                np.ix_(indices, indices)
            ]
            mean_dists = submat.mean(axis=1)
            cluster_rep_idx = indices[np.argmin(mean_dists).item()]
            cluster_rep = acc_ordering[cluster_rep_idx]
            print(
                path_to_seq(cluster_rep, index_df),
                file=out_file
            )

    print("Done.")


if __name__ == "__main__":
    main()
