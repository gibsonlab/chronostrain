import itertools
from pathlib import Path
from typing import List, Dict, Iterator, Tuple
import argparse

import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sb
import ot
from Bio import SeqIO

from chronostrain.database import StrainDatabase
from chronostrain import cfg


def read_depth_dirs(base_dir: Path) -> Iterator[Tuple[int, Path]]:
    for child_dir in base_dir.glob("reads_*"):
        if not child_dir.is_dir():
            raise RuntimeError(f"Expected child `{child_dir}` to be a directory.")

        read_depth = int(child_dir.name.split("_")[1])
        yield read_depth, child_dir


def trial_dirs(read_depth_dir: Path) -> Iterator[Tuple[int, Path]]:
    for child_dir in read_depth_dir.glob("trial_*"):
        if not child_dir.is_dir():
            raise RuntimeError(f"Expected child `{child_dir}` to be a directory.")

        trial_num = int(child_dir.name.split("_")[1])
        yield trial_num, child_dir


def load_ground_truth(ground_truth_path: Path) -> pd.DataFrame:
    df_entries = []
    with open(ground_truth_path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        header_row = next(reader)

        assert header_row[0] == 'T'
        strain_ids = header_row[1:]
        for row in reader:
            t = float(row[0])
            for strain_id, abund in zip(strain_ids, row[1:]):
                abund = float(abund)
                df_entries.append({'T': t, 'Strain': strain_id, 'RelAbund': abund})
    return pd.DataFrame(df_entries)


def hamming_distance(x: str, y: str) -> int:
    assert len(x) == len(y)
    return sum(1 for c, d in zip(x, y) if c != d)


def parse_hamming(strain_ids: List[str], multi_align_path: Path) -> np.ndarray:
    """Use pre-constructed multiple alignment to compute distance in hamming space."""
    aligned_seqs: List[str] = ['' for _ in strain_ids]
    strain_idxs: Dict[str, int] = {sid: i for i, sid in enumerate(strain_ids)}
    for record in SeqIO.parse(multi_align_path, 'fasta'):
        strain_id = record.id
        if strain_id not in strain_idxs:
            continue
        strain_idx = strain_idxs[strain_id]
        aligned_seqs[strain_idx] = str(record.seq)

    matrix = np.zeros(
        shape=(len(strain_ids), len(strain_ids)),
        dtype=float
    )
    for (i, i_seq), (j, j_seq) in itertools.combinations(enumerate(aligned_seqs), r=2):
        d = hamming_distance(i_seq, j_seq)
        matrix[i, j] = d
        matrix[j, i] = d
    return matrix


def parse_chronostrain_estimate(db: StrainDatabase,
                                ground_truth: pd.DataFrame,
                                strain_ids: List[str],
                                output_dir: Path) -> np.ndarray:
    samples = torch.load(output_dir / 'samples.pt')
    strains = db.all_strains()

    time_points = sorted(pd.unique(ground_truth['T']))

    if samples.shape[0] != len(time_points):
        raise RuntimeError("Number of time points ({}) in ground truth don't match sampled time points ({}).".format(
            len(time_points),
            samples.shape[0]
        ))

    if samples.shape[2] != len(strains):
        raise RuntimeError("Number of strains ({}) in database don't match sampled strain counts ({}).".format(
            len(strains),
            samples.shape[2]
        ))

    inferred_abundances = torch.softmax(samples, dim=2)
    median_abundances = np.median(inferred_abundances.numpy(), axis=1)

    estimate = np.zeros(shape=(len(time_points), len(strain_ids)), dtype=float)
    strain_indices = {sid: i for i, sid in enumerate(strain_ids)}
    for db_idx, strain in enumerate(strains):
        s_idx = strain_indices[strain.id]
        estimate[:, s_idx] = median_abundances[:, db_idx]
    return estimate


def parse_strainest_estimate(ground_truth: pd.DataFrame,
                             strain_ids: List[str],
                             output_dir: Path) -> np.ndarray:
    time_points = sorted(pd.unique(ground_truth['T']))
    strain_indices = {sid: i for i, sid in enumerate(strain_ids)}

    est_rel_abunds = np.zeros(shape=(len(time_points), len(strain_ids)), dtype=float)
    for t_idx, t in enumerate(time_points):
        output_path = output_dir / f"abund_{t_idx}.txt"
        if not output_path.exists():
            continue

        with open(output_path, 'rt') as f:
            lines = iter(f)
            header_line = next(lines)
            if not header_line.startswith('OTU'):
                raise RuntimeError(f"Unexpected format for file `{output_path}` generated by StrainEst.")

            for line in lines:
                strain_id, abund = line.rstrip().split('\t')
                abund = float(abund)
                strain_idx = strain_indices[strain_id]
                est_rel_abunds[t_idx][strain_idx] = abund

    return est_rel_abunds


def wasserstein_error(abundance_est: np.ndarray, truth_df: pd.DataFrame, strain_distances: np.ndarray, strain_ids: List[str]) -> float:
    time_points = sorted(pd.unique(truth_df['T']))
    ground_truth = np.zeros(shape=(len(time_points), len(strain_ids)), dtype=float)

    t_idxs = {t: t_idx for t_idx, t in enumerate(time_points)}
    strain_idxs = {sid: i for i, sid in enumerate(strain_ids)}

    for _, row in truth_df.iterrows():
        s_idx = strain_idxs[row['Strain']]
        t_idx = t_idxs[row['T']]
        ground_truth[t_idx, s_idx] = row['RelAbund']

    return sum(
        compute_wasserstein(ground_truth[t_idx], abundance_est[t_idx], strain_distances, verbose=False)
        for t_idx in range(len(time_points))
    )


def compute_wasserstein(
        abund1: np.ndarray,
        abund2: np.ndarray,
        distance_matrix: np.ndarray,
        verbose: bool = False
) -> float:
    """Computes the wasserstein distance. A simple wrapper around `ot.sinkhorn` call with default lambda value."""
    lambd = 1e-3
    return ot.sinkhorn(
        abund1,
        abund2,
        distance_matrix,
        lambd,
        verbose=verbose
    )


def all_ecoli_strain_ids(index_path: Path) -> List[str]:
    df = pd.read_csv(index_path, sep='\t')
    return list(pd.unique(df.loc[
        (df['Genus'] == 'Escherichia') & (df['Species'] == 'coli'),
        'Accession'
    ]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index_file', type=str, required=True)
    parser.add_argument('-a', '--alignment_file', type=str, required=True)
    parser.add_argument('-v', '--verbose', type=bool, action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_data_dir)
    out_path = Path(args.out_path)

    # Necessary precomputation.
    ground_truth = load_ground_truth(Path(args.ground_truth_path))
    db = cfg.database_cfg.get_database()
    strain_ids = all_ecoli_strain_ids(Path(args.index_file))
    distances = parse_hamming(strain_ids, Path(args.alignment_file))

    # search through all of the read depths.
    df_entries = []
    for read_depth, read_depth_dir in read_depth_dirs(base_dir):
        for trial_num, trial_dir in trial_dirs(read_depth_dir):
            print(f"Handling read depth {read_depth}, trial {trial_num}")

            # =========== Chronostrain
            chronostrain_estimate = parse_chronostrain_estimate(db, ground_truth, strain_ids, trial_dir / 'output' / 'chronostrain')
            df_entries.append({
                'ReadDepth': read_depth,
                'TrialNum': trial_num,
                'Method': 'Chronostrain',
                'Error': wasserstein_error(chronostrain_estimate, ground_truth, distances, strain_ids)
            })

            # =========== StrainEst
            strainest_estimate = parse_strainest_estimate(ground_truth, strain_ids, trial_dir / 'output' / 'strainest')
            df_entries.append({
                'ReadDepth': read_depth,
                'TrialNum': trial_num,
                'Method': 'StrainEst',
                'Error': wasserstein_error(strainest_estimate, ground_truth, distances, strain_ids)
            })

    summary_df = pd.DataFrame(df_entries)
    summary_df.to_csv(out_path, index=False)
    print(f"[*] Saved results to {out_path}.")

    plot_path = out_path.parent / "plot.pdf"
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    sb.boxplot(
        data=summary_df,
        x='ReadDepth',
        hue='Method',
        y='Error',
        ax=ax
    )

    plt.savefig(plot_path)
    print(f"[*] Saved plot to {plot_path}.")