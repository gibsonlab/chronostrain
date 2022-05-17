import itertools
from pathlib import Path
from typing import List, Iterator, Tuple
import argparse
import pickle

import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sb
import ot
from Bio import SeqIO
from tqdm import tqdm

from chronostrain.config import create_logger
from chronostrain.database import StrainDatabase
from chronostrain import cfg

logger = create_logger("chronostrain.evaluate")
device = torch.device("cuda:0")


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


def parse_hamming(multi_align_path: Path, index_df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Use pre-constructed multiple alignment to compute distance in hamming space."""
    strain_ids: List[str] = []
    aligned_seqs: List[str] = []
    for record in SeqIO.parse(multi_align_path, 'fasta'):
        strain_id = record.id

        if index_df.loc[index_df['Accession'] == strain_id, 'Species'].item() != 'coli':
            continue

        strain_ids.append(strain_id)
        aligned_seqs.append(str(record.seq))

    logger.info("Found {} strains.".format(len(strain_ids)))
    matrix = np.zeros(
        shape=(len(strain_ids), len(strain_ids)),
        dtype=float
    )

    n_strains = len(strain_ids)
    n_pairs = int(n_strains * (n_strains - 1)) // 2
    for (i, i_seq), (j, j_seq) in tqdm(itertools.combinations(enumerate(aligned_seqs), r=2), total=n_pairs):
        if len(i_seq) != len(j_seq):
            raise RuntimeError("Found mismatching string lengths {} and {} (strains {} vs {})".format(
                len(i_seq), len(j_seq),
                strain_ids[i], strain_ids[j]
            ))
        d = hamming_distance(i_seq, j_seq)
        matrix[i, j] = d
        matrix[j, i] = d
    return strain_ids, matrix


def strip_suffixes(strain_id_string: str):
    suffixes = {'.chrom', '.fna', '.gz', '.bz', '.fastq', '.fasta'}
    x = Path(strain_id_string)
    while x.suffix in suffixes:
        x = x.with_suffix('')
    return x.name


def parse_chronostrain_estimate(db: StrainDatabase,
                                ground_truth: pd.DataFrame,
                                strain_ids: List[str],
                                output_dir: Path) -> torch.Tensor:
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
    n_samples = samples.size(1)
    estimate = torch.zeros(size=(len(time_points), n_samples, len(strain_ids)), dtype=torch.float, device=device)
    strain_indices = {sid: i for i, sid in enumerate(strain_ids)}
    for db_idx, strain in enumerate(strains):
        s_idx = strain_indices[strain.id]
        estimate[:, :, s_idx] = inferred_abundances[:, :, db_idx]
    return estimate


def parse_strainest_estimate(ground_truth: pd.DataFrame,
                             strain_ids: List[str],
                             output_dir: Path) -> torch.Tensor:
    time_points = sorted(pd.unique(ground_truth['T']))
    strain_indices = {sid: i for i, sid in enumerate(strain_ids)}

    est_rel_abunds = torch.zeros(size=(len(time_points), len(strain_ids)), dtype=torch.float, device=device)
    for t_idx, t in enumerate(time_points):
        output_path = output_dir / f"abund_{t_idx}.txt"
        with open(output_path, 'rt') as f:
            lines = iter(f)
            header_line = next(lines)
            if not header_line.startswith('OTU'):
                raise RuntimeError(f"Unexpected format for file `{output_path}` generated by StrainEst.")

            for line in lines:
                strain_id, abund = line.rstrip().split('\t')
                strain_id = strip_suffixes(strain_id)
                abund = float(abund)
                try:
                    strain_idx = strain_indices[strain_id]
                    est_rel_abunds[t_idx][strain_idx] = abund
                except KeyError as e:
                    continue

    # Renormalize.
    row_sum = torch.sum(est_rel_abunds, dim=1, keepdim=True)
    support = torch.where(row_sum > 0)[0]
    zeros = torch.where(row_sum == 0)[0]
    est_rel_abunds[support, :] = est_rel_abunds[support, :] / row_sum[support]
    est_rel_abunds[zeros, :] = 1 / len(strain_ids)
    return est_rel_abunds


def parse_straingst_estimate(
        ground_truth: pd.DataFrame,
        strain_ids: List[str],
        output_dir: Path
) -> torch.Tensor:
    time_points = sorted(pd.unique(ground_truth['T']))
    strain_indices = {strain_id: s_idx for s_idx, strain_id in enumerate(strain_ids)}

    est_rel_abunds = torch.zeros(size=(len(time_points), len(strain_ids)), dtype=torch.float, device=device)
    for t_idx, t in enumerate(time_points):
        output_path = output_dir / f"output_mash_{t_idx}.tsv"
        if not output_path.exists():
            continue
        with open(output_path, 'r') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            _ = next(reader)
            _ = next(reader)
            line3 = next(reader)
            assert line3[0] == 'i'

            for row in reader:
                strain_id = row[1]
                strain_id = strip_suffixes(strain_id)
                strain_idx = strain_indices[strain_id]
                rel_abund = float(row[11]) / 100.0
                est_rel_abunds[t_idx][strain_idx] = rel_abund

    # Renormalize.
    row_sum = torch.sum(est_rel_abunds, dim=1, keepdim=True)
    support = torch.where(row_sum > 0)[0]
    zeros = torch.where(row_sum == 0)[0]
    est_rel_abunds[support, :] = est_rel_abunds[support, :] / row_sum[support]
    est_rel_abunds[zeros, :] = 1 / len(strain_ids)
    return est_rel_abunds


def wasserstein_error(abundance_est: torch.Tensor, truth_df: pd.DataFrame, strain_distances: torch.Tensor, strain_ids: List[str]) -> torch.Tensor:
    time_points = sorted(pd.unique(truth_df['T']))
    ground_truth = torch.zeros(size=(len(time_points), len(strain_ids)), dtype=torch.float, device=device)

    t_idxs = {t: t_idx for t_idx, t in enumerate(time_points)}
    strain_idxs = {sid: i for i, sid in enumerate(strain_ids)}

    for _, row in truth_df.iterrows():
        s_idx = strain_idxs[row['Strain']]
        t_idx = t_idxs[row['T']]
        ground_truth[t_idx, s_idx] = row['RelAbund']

    if len(abundance_est.shape) == 2:
        answers = torch.cat([
            compute_wasserstein(ground_truth[t_idx], abundance_est[t_idx].unsqueeze(1), strain_distances)
            for t_idx in range(len(time_points))
        ], dim=0)
        return answers.sum()
    elif len(abundance_est.shape) == 3:
        w_errors = torch.stack([
            compute_wasserstein(ground_truth[t_idx], torch.transpose(abundance_est[t_idx, :, ], 0, 1), strain_distances)
            for t_idx in range(len(time_points))
        ], dim=0)
        return w_errors.sum(dim=0)
    else:
        raise ValueError("Cannot handle abundance estimate matrices of dimension != (2 or 3).")


def compute_wasserstein(
        src_histogram: torch.Tensor,
        tgt_histogram: torch.Tensor,
        distance_matrix: torch.Tensor
) -> torch.Tensor:
    """Computes the wasserstein distance. A simple wrapper around `ot.sinkhorn` call with default regularization value."""
    wasserstein = ot.sinkhorn(
        src_histogram,
        tgt_histogram,
        distance_matrix,
        verbose=False,
        reg=1e-2,
        method='sinkhorn_log',
        numItermax=500
    )
    return wasserstein


def all_ecoli_strain_ids(index_path: Path) -> List[str]:
    df = pd.read_csv(index_path, sep='\t')
    return list(pd.unique(df.loc[
        (df['Genus'] == 'Escherichia') & (df['Species'] == 'coli'),
        'Accession'
    ]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_data_dir', type=str, required=True)
    parser.add_argument('-i', '--index_path', type=str, required=True)
    parser.add_argument('-a', '--alignment_file', type=str, required=True)
    parser.add_argument('-o', '--out_dir', type=str, required=True)
    parser.add_argument('-g', '--ground_truth_path', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.base_data_dir)
    out_dir = Path(args.out_dir)

    # Necessary precomputation.
    ground_truth = load_ground_truth(Path(args.ground_truth_path))
    index_df = pd.read_csv(args.index_path, sep='\t')
    chronostrain_db = cfg.database_cfg.get_database()
    out_dir.mkdir(exist_ok=True, parents=True)

    dists_path = out_dir / 'strain_distances.pkl'
    try:
        with open(dists_path, 'rb') as f:
            strain_ids = pickle.load(f)
            distances = torch.tensor(pickle.load(f), device=device)
    except BaseException:
        logger.info("Parsing hamming distances.")
        strain_ids, distances = parse_hamming(Path(args.alignment_file), index_df)
        distances = torch.tensor(distances, device=device)
        with open(dists_path, 'wb') as f:
            pickle.dump(strain_ids, f)
            pickle.dump(distances, f)

    # search through all of the read depths.
    df_entries = []
    for read_depth, read_depth_dir in read_depth_dirs(base_dir):
        for trial_num, trial_dir in trial_dirs(read_depth_dir):
            logger.info(f"Handling read depth {read_depth}, trial {trial_num}")

            # =========== Chronostrain
            try:
                logger.info("Computing chronostrain error...")
                chronostrain_estimate_samples = parse_chronostrain_estimate(chronostrain_db, ground_truth, strain_ids,
                                                                            trial_dir / 'output' / 'chronostrain')
                errors = wasserstein_error(
                    chronostrain_estimate_samples[:, :30, :],
                    ground_truth, distances, strain_ids
                )
                logger.info("Chronostrain Mean error: {}".format(errors.mean()))
                for sample_idx in range(len(errors)):
                    df_entries.append({
                        'ReadDepth': read_depth,
                        'TrialNum': trial_num,
                        'SampleIdx': sample_idx,
                        'Method': 'Chronostrain',
                        'Error': errors[sample_idx]
                    })
            except FileNotFoundError:
                logger.info("Skipping Chronostrain output.")

            # =========== StrainEst
            try:
                strainest_estimate = parse_strainest_estimate(ground_truth, strain_ids,
                                                              trial_dir / 'output' / 'strainest')
                error = wasserstein_error(strainest_estimate, ground_truth, distances, strain_ids).item()
                logger.info("StrainEst Error: {}".format(error))
                df_entries.append({
                    'ReadDepth': read_depth,
                    'TrialNum': trial_num,
                    'SampleIdx': 0,
                    'Method': 'StrainEst',
                    'Error': error
                })
            except FileNotFoundError:
                logger.info("Skipping StrainEst output.")

            # =========== StrainGST
            try:
                straingst_estimate = parse_straingst_estimate(ground_truth, strain_ids,
                                                              trial_dir / 'output' / 'straingst')
                error = wasserstein_error(straingst_estimate, ground_truth, distances, strain_ids).item()
                logger.info("StrainGST Error: {}".format(error))
                df_entries.append({
                    'ReadDepth': read_depth,
                    'TrialNum': trial_num,
                    'SampleIdx': 0,
                    'Method': 'StrainGST',
                    'Error': error
                })
            except FileNotFoundError:
                logger.info("Skipping StrainGST output.")

    out_path = out_dir / 'summary.csv'
    summary_df = pd.DataFrame(df_entries)
    summary_df.to_csv(out_path, index=False)
    logger.info(f"[*] Saved results to {out_path}.")

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
    logger.info(f"[*] Saved plot to {plot_path}.")


if __name__ == "__main__":
    main()
