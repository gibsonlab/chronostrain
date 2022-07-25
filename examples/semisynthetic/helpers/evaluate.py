import itertools
from pathlib import Path
from typing import List, Iterator, Tuple
import argparse

import csv
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
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
    abundance_samples = torch.load(output_dir / 'samples.pt')
    db_strains = [s.id for s in db.all_strains()]

    time_points = sorted(pd.unique(ground_truth['T']))
    if abundance_samples.shape[0] != len(time_points):
        raise RuntimeError("Number of time points ({}) in ground truth don't match sampled time points ({}).".format(
            len(time_points),
            abundance_samples.shape[0]
        ))

    if abundance_samples.shape[2] != len(db_strains):
        raise RuntimeError("Number of strains ({}) in database don't match sampled strain counts ({}).".format(
            len(db_strains),
            abundance_samples.shape[2]
        ))

    n_samples = abundance_samples.size(1)
    estimate = torch.zeros(size=(len(time_points), n_samples, len(strain_ids)), dtype=torch.float, device=device)
    strain_indices = {sid: i for i, sid in enumerate(strain_ids)}
    for db_idx, strain_id in enumerate(db_strains):
        s_idx = strain_indices[strain_id]
        estimate[:, :, s_idx] = abundance_samples[:, :, db_idx]

    # Renormalize.
    sums = torch.sum(estimate, dim=-1, keepdim=True)
    estimate = estimate / sums
    return estimate


def parse_strainest_estimate(ground_truth: pd.DataFrame,
                             strain_ids: List[str],
                             sensitivity: str,
                             output_dir: Path) -> torch.Tensor:
    time_points = sorted(pd.unique(ground_truth['T']))
    strain_indices = {sid: i for i, sid in enumerate(strain_ids)}

    est_rel_abunds = torch.zeros(size=(len(time_points), len(strain_ids)), dtype=torch.float, device=device)
    for t_idx, t in enumerate(time_points):
        output_path = output_dir / f"abund_{t_idx}.{sensitivity}.txt"
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
        output_dir: Path,
        mode: str
) -> torch.Tensor:
    time_points = sorted(pd.unique(ground_truth['T']))
    strain_indices = {strain_id: s_idx for s_idx, strain_id in enumerate(strain_ids)}

    est_rel_abunds = torch.zeros(size=(len(time_points), len(strain_ids)), dtype=torch.float, device=device)
    for t_idx, t in enumerate(time_points):
        output_path = output_dir / mode / f"output_mash_{t_idx}.tsv"
        with open(output_path, 'r') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            _ = next(reader)
            _ = next(reader)
            line3 = next(reader)
            assert line3[0] == 'i'

            for row in reader:
                strain_id = row[1]
                strain_id = strip_suffixes(strain_id)

                if strain_id not in strain_indices:
                    continue

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


def parse_strainfacts_estimate(
        truth_df: pd.DataFrame,
        strain_ids: List[str],
        output_dir: Path
) -> torch.Tensor:
    time_points = sorted(pd.unique(truth_df['T']))
    supported_strains = list(pd.unique(truth_df['Strain']))
    ground_truth = extract_ground_truth_array(truth_df, supported_strains)

    est_rel_abunds = torch.zeros(size=(len(time_points), len(supported_strains)), dtype=torch.float, device=device)
    with open(output_dir / 'result_community.tsv', 'r') as f:
        for line in f:
            if line.startswith("sample"):
                continue

            t_idx, s_idx, abnd = line.rstrip().split('\t')
            t_idx = int(t_idx)
            s_idx = int(s_idx)
            abnd = float(abnd)
            if s_idx >= len(supported_strains):
                raise ValueError("Didn't expect more than {} strains in output of StrainFacts.".format(len(supported_strains)))
            est_rel_abunds[t_idx, s_idx] = abnd

    # Compute minimal permutation.
    minimal_error = float("inf")
    best_perm = tuple(range(len(supported_strains)))
    for perm in itertools.permutations(list(range(len(supported_strains)))):
        permuted_est = est_rel_abunds[:, perm]
        perm_error = error_metric(permuted_est, ground_truth)
        if perm_error < minimal_error:
            minimal_error = perm_error
            best_perm = perm

    all_idxs = {s: i for i, s in enumerate(strain_ids)}
    support_idx = [all_idxs[s] for s in supported_strains]
    full_est = torch.zeros(size=(len(time_points), len(strain_ids)), dtype=torch.float, device=device)
    full_est[:, support_idx] = est_rel_abunds[:, best_perm]
    return full_est


def extract_ground_truth_array(truth_df: pd.DataFrame, strain_ids: List[str]) -> torch.Tensor:
    time_points = sorted(pd.unique(truth_df['T']))
    t_idxs = {t: t_idx for t_idx, t in enumerate(time_points)}
    strain_idxs = {sid: i for i, sid in enumerate(strain_ids)}
    ground_truth = torch.zeros(size=(len(time_points), len(strain_ids)), dtype=torch.float, device=device)
    for _, row in truth_df.iterrows():
        s_idx = strain_idxs[row['Strain']]
        t_idx = t_idxs[row['T']]
        ground_truth[t_idx, s_idx] = row['RelAbund']
    return ground_truth


def error_metric(abundance_est: torch.Tensor, truth: torch.Tensor) -> float:
    assert len(abundance_est.shape) == 2

    _T = abundance_est.shape[0]
    _S = abundance_est.shape[1]

    l1_error = torch.sum(torch.abs(truth - abundance_est))
    return l1_error.item()


def engraftment_ratio(presence: torch.Tensor) -> float:
    """
    :param presence: (T x S) Tensor of boolean values.
    :return:
    """
    t1 = presence[:-1]
    t2 = presence[1:]
    if torch.sum(t1).item() == 0:
        return float('inf')
    else:
        return torch.sum(torch.logical_and(t1, t2)).item() / torch.sum(t1).item()


def clearance_ratio(presence: torch.Tensor) -> float:
    """
    :param presence: (T x S) Tensor of boolean values.
    :return:
    """
    return engraftment_ratio(torch.logical_not(presence))


def chronostrain_presence(abundance_est: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """
    :param abundance_est: (T x N x S) tensor of abundance samples.
    :param q: the quantile to compute.
    :return:
    """
    lb = 1 / abundance_est.shape[-1]
    return torch.quantile(abundance_est, q, dim=1) > lb


def other_method_presence(abundance_est: torch.Tensor) -> torch.Tensor:
    return abundance_est != 0


def dominance_coeff(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    :param x: a 1-d Tensor of values.
    :param y: a 1-d Tensor of values of the same size as x.
    :param eps: A padding value to prevent division by zeroes.
    :return: the entrywise dominance coefficient log(x) - log(y).
    """
    return torch.divide(x + eps, y + eps)


def dominance_error(abundance_est: torch.Tensor, truth: torch.Tensor, strain_ids: List[str], strain1: str, strain2: str) -> float:
    idxs = {sid: i for i, sid in enumerate(strain_ids)}
    sidx1 = idxs[strain1]
    sidx2 = idxs[strain2]

    _est = dominance_coeff(abundance_est[:, sidx1], abundance_est[:, sidx2])
    _truth = dominance_coeff(truth[:, sidx1], truth[:, sidx2])
    return torch.exp(
        torch.mean(torch.log(_est) - torch.log(_truth))
    ).item()  # geometric mean


# def wasserstein_error(abundance_est: torch.Tensor, truth_df: pd.DataFrame, strain_distances: torch.Tensor, strain_ids: List[str]) -> torch.Tensor:
#     time_points = sorted(pd.unique(truth_df['T']))
#     ground_truth = extract_ground_truth_array(truth_df, strain_ids)
#
#     if len(abundance_est.shape) == 2:
#         answers = torch.cat([
#             compute_wasserstein(ground_truth[t_idx], abundance_est[t_idx].unsqueeze(1), strain_distances)
#             for t_idx in range(len(time_points))
#         ], dim=0)
#         return answers.sum()
#     elif len(abundance_est.shape) == 3:
#         w_errors = torch.stack([
#             compute_wasserstein(ground_truth[t_idx], torch.transpose(abundance_est[t_idx, :, ], 0, 1), strain_distances)
#             for t_idx in range(len(time_points))
#         ], dim=0)
#         return w_errors.sum(dim=0)
#     else:
#         raise ValueError("Cannot handle abundance estimate matrices of dimension != (2 or 3).")


# def compute_wasserstein(
#         src_histogram: torch.Tensor,
#         tgt_histogram: torch.Tensor,
#         distance_matrix: torch.Tensor
# ) -> torch.Tensor:
#     """Computes the wasserstein distance. A simple wrapper around `ot.sinkhorn` call with default regularization value."""
#     wasserstein = ot.sinkhorn(
#         src_histogram,
#         tgt_histogram,
#         distance_matrix,
#         verbose=False,
#         reg=1e-2,
#         method='sinkhorn_log',
#         numItermax=500
#     )
#     return wasserstein


def all_ecoli_strain_ids(index_path: Path) -> List[str]:
    df = pd.read_csv(index_path, sep='\t')
    return list(pd.unique(df.loc[
        (df['Genus'] == 'Escherichia') & (df['Species'] == 'coli'),
        'Accession'
    ]))


def plot_result(out_path: Path, truth_df: pd.DataFrame, samples: torch.Tensor, strain_ordering: List[str]):
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    viridis = matplotlib.cm.get_cmap('viridis', len(strain_ordering))
    cmap = viridis(np.linspace(0, 1, len(strain_ordering)))

    q_lower = 0.025
    q_upper = 0.975
    t = sorted(float(x) for x in pd.unique(truth_df['T']))
    time_points = sorted(pd.unique(truth_df['T']))
    t_idxs = {t: t_idx for t_idx, t in enumerate(time_points)}
    strain_idxs = {sid: i for i, sid in enumerate(strain_ordering)}
    ground_truth = torch.zeros(size=(len(time_points), len(strain_ordering)), dtype=torch.float, device=device)
    for _, row in truth_df.iterrows():
        s_idx = strain_idxs[row['Strain']]
        t_idx = t_idxs[row['T']]
        ground_truth[t_idx, s_idx] = row['RelAbund']

    if len(samples.shape) == 3:
        for s_idx, strain_id in enumerate(strain_ordering):
            traj = samples[:, :, s_idx]
            truth_traj = ground_truth[:, s_idx].cpu().numpy()
            lower = torch.quantile(traj, q_lower, dim=1).cpu().numpy()
            upper = torch.quantile(traj, q_upper, dim=1).cpu().numpy()
            median = torch.median(traj, dim=1).values.cpu().numpy()

            color = cmap[s_idx]
            ax.fill_between(t, lower, upper, alpha=0.3, color=color)
            ax.plot(t, median, color=color)
            ax.plot(t, truth_traj, color=color, linestyle='dashed')
    elif len(samples.shape) == 2:
        for s_idx, strain_id in enumerate(strain_ordering):
            traj = samples[:, s_idx].cpu().numpy()
            truth_traj = ground_truth[:, s_idx].cpu().numpy()
            color = cmap[s_idx]
            ax.plot(t, traj, color=color)
            ax.plot(t, truth_traj, color=color, linestyle='dashed')
    else:
        raise RuntimeError(f"Can't plot samples of dimension {len(samples.shape)}")
    plt.savefig(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_data_dir', type=str, required=True)
    parser.add_argument('-i', '--index_path', type=str, required=True)
    parser.add_argument('-a', '--alignment_file', type=str, required=True)
    parser.add_argument('-o', '--out_dir', type=str, required=True)
    parser.add_argument('-g', '--ground_truth_path', type=str, required=True)
    return parser.parse_args()


def evaluate_errors(ground_truth: pd.DataFrame,
                    result_base_dir: Path,
                    chronostrain_db: StrainDatabase) -> pd.DataFrame:
    strain_ids = list(pd.unique(ground_truth.loc[ground_truth['RelAbund'] > 0, 'Strain']))
    truth_tensor = extract_ground_truth_array(ground_truth, strain_ids)

    def engraftment_clearance(pres: torch.Tensor) -> Tuple[float, float]:
        return engraftment_ratio(pres), clearance_ratio(pres)

    # search through all of the read depths.
    df_entries = []
    for read_depth, read_depth_dir in read_depth_dirs(result_base_dir):
        for trial_num, trial_dir in trial_dirs(read_depth_dir):
            logger.info(f"Handling read depth {read_depth}, trial {trial_num}")
            plot_dir = trial_dir / 'output' / 'plots'
            plot_dir.mkdir(exist_ok=True, parents=True)

            # =========== Chronostrain
            try:
                chronostrain_estimate_samples = parse_chronostrain_estimate(chronostrain_db, ground_truth, strain_ids,
                                                                            trial_dir / 'output' / 'chronostrain' / 'full_corr')
                # errors = wasserstein_error(
                #     chronostrain_estimate_samples[:, :30, :],
                #     ground_truth, distances, strain_ids
                # )
                error = error_metric(torch.median(chronostrain_estimate_samples, dim=1).values, truth_tensor)
                engraftment, clearance = engraftment_clearance(chronostrain_presence(chronostrain_estimate_samples))
                dom_err = dominance_error(torch.median(chronostrain_estimate_samples, dim=1).values, truth_tensor, strain_ids, "NZ_CP069709.1", "NZ_CP076645.1")

                logger.info("Chronostrain: err = {}, engraft = {}, clear = {}".format(error, engraftment, clearance))

                df_entries.append({
                    'ReadDepth': read_depth,
                    'TrialNum': trial_num,
                    'Method': 'Chronostrain',
                    'Error': error,
                    'Engraftment': engraftment,
                    'Clearance': clearance,
                    'Dominance': dom_err
                })

                plot_result(plot_dir / 'chronostrain.pdf', ground_truth, chronostrain_estimate_samples, strain_ids)
            except FileNotFoundError:
                logger.info("Skipping Chronostrain output.")

            # =========== StrainEst (Sensitive)
            try:
                strainest_sens_estimate = parse_strainest_estimate(ground_truth, strain_ids,
                                                                   'sensitive',
                                                                   trial_dir / 'output' / 'strainest')
                # error = wasserstein_error(strainest_estimate, ground_truth, distances, strain_ids).item()
                error = error_metric(strainest_sens_estimate, truth_tensor)
                engraftment, clearance = engraftment_clearance(other_method_presence(strainest_sens_estimate))
                dom_err = dominance_error(strainest_sens_estimate, truth_tensor, strain_ids, "NZ_CP069709.1", "NZ_CP076645.1")

                logger.info("StrainEst (Sens) err = {}, engraft = {}, clear = {}".format(error, engraftment, clearance))
                df_entries.append({
                    'ReadDepth': read_depth,
                    'TrialNum': trial_num,
                    'Method': 'StrainEst (Sensitive)',
                    'Error': error,
                    'Engraftment': engraftment,
                    'Clearance': clearance,
                    'Dominance': dom_err
                })
                plot_result(plot_dir / 'strainest.sensitive.pdf', ground_truth, strainest_sens_estimate, strain_ids)
            except FileNotFoundError:
                logger.info("Skipping StrainEst (Sensitive) output.")

            # =========== StrainEst (Default)
            try:
                strainest_estimate = parse_strainest_estimate(ground_truth, strain_ids,
                                                              'default',
                                                              trial_dir / 'output' / 'strainest')
                error = error_metric(strainest_estimate, truth_tensor)
                engraftment, clearance = engraftment_clearance(other_method_presence(strainest_estimate))
                dom_err = dominance_error(strainest_estimate, truth_tensor, strain_ids, "NZ_CP069709.1", "NZ_CP076645.1")

                logger.info("StrainEst (Default) err = {}, engraft = {}, clear = {}".format(error, engraftment, clearance))
                df_entries.append({
                    'ReadDepth': read_depth,
                    'TrialNum': trial_num,
                    'Method': 'StrainEst (Default)',
                    'Error': error,
                    'Engraftment': engraftment,
                    'Clearance': clearance,
                    'Dominance': dom_err
                })
                plot_result(plot_dir / 'strainest.default.pdf', ground_truth, strainest_estimate, strain_ids)
            except FileNotFoundError:
                logger.info("Skipping StrainEst (Default) output.")

            # =========== StrainGST (whole genome)
            try:
                straingst_estimate = parse_straingst_estimate(ground_truth, strain_ids,
                                                              trial_dir / 'output' / 'straingst',
                                                              mode='chromosome')
                # error = wasserstein_error(straingst_estimate, ground_truth, distances, strain_ids).item()
                error = error_metric(straingst_estimate, truth_tensor)
                engraftment, clearance = engraftment_clearance(other_method_presence(straingst_estimate))
                dom_err = dominance_error(straingst_estimate, truth_tensor, strain_ids, "NZ_CP069709.1", "NZ_CP076645.1")
                logger.info("StrainGST err = {}, engraft = {}, clear = {}".format(error, engraftment, clearance))
                df_entries.append({
                    'ReadDepth': read_depth,
                    'TrialNum': trial_num,
                    'Method': 'StrainGST',
                    'Error': error,
                    'Engraftment': engraftment,
                    'Clearance': clearance,
                    'Dominance': dom_err
                })
                plot_result(plot_dir / 'straingst.pdf', ground_truth, straingst_estimate, strain_ids)
            except FileNotFoundError:
                logger.info("Skipping StrainGST output.")

            # # =========== StrainGST (markers)
            # try:
            #     straingst_estimate = parse_straingst_estimate(ground_truth, strain_ids,
            #                                                   trial_dir / 'output' / 'straingst',
            #                                                   mode='markers')
            #     # error = wasserstein_error(straingst_estimate, ground_truth, distances, strain_ids).item()
            #     error = error_metric(straingst_estimate, truth_tensor)
            #     logger.info("StrainGST Error: {}".format(error))
            #     df_entries.append({
            #         'ReadDepth': read_depth,
            #         'TrialNum': trial_num,
            #         'Method': 'StrainGST (Markers)',
            #         'Error': error
            #     })
            #     # plot_result(plot_dir / 'straingst_marker.pdf', ground_truth, straingst_estimate, strain_ids)
            # except FileNotFoundError:
            #     logger.info("Skipping StrainGST (markers) output.")

            # =========== StrainFacts
            try:
                strainfacts_estimate = parse_strainfacts_estimate(ground_truth,
                                                                  strain_ids,
                                                                  trial_dir / 'output' / 'strainfacts')
                error = error_metric(strainfacts_estimate, truth_tensor)
                dom_err = dominance_error(strainfacts_estimate, truth_tensor, strain_ids, "NZ_CP069709.1", "NZ_CP076645.1")
                logger.info("StrainFacts Error: {}".format(error))
                df_entries.append({
                    'ReadDepth': read_depth,
                    'TrialNum': trial_num,
                    'Method': 'StrainFacts',
                    'Error': error,
                    'Engraftment': float('inf'),
                    'Clearance': float('inf'),
                    'Dominance': dom_err
                })
                # plot_result(plot_dir / 'strainfacts.pdf', ground_truth, strainfacts_estimate, strain_ids)
            except FileNotFoundError:
                logger.info("Skipping StrainFacts output.")

    return pd.DataFrame(df_entries)


def evaluate_runtimes(result_base_dir: Path):
    df_entries = []

    for read_depth, read_depth_dir in read_depth_dirs(result_base_dir):
        for trial_num, trial_dir in trial_dirs(read_depth_dir):
            def parse_runtime_file(_method_name: str, _method_part: str, _time_point: str, _runtime_file: str):
                runtime_path = trial_dir / "output" / _runtime_file
                if runtime_path.exists():
                    with open(runtime_path, 'rt') as f:
                        duration = int(next(iter(f)))
                        df_entries.append({
                            'Method': _method_name,
                            'MethodPart': _method_part,
                            'ReadDepth': read_depth,
                            'Trial': trial_num,
                            'Timepoint': _time_point,
                            'Duration': duration
                        })
                else:
                    logger.debug(f"Skipping {runtime_path}")

            logger.info(f"Handling read depth {read_depth}, trial {trial_num}")
            parse_runtime_file('Chronostrain', 'Filter', 'all', 'chronostrain_filter_runtime.txt')
            parse_runtime_file('Chronostrain', 'Inference', 'all', 'chronostrain_runtime.txt')
            parse_runtime_file('StrainGST', 'all', '0', 'straingst_runtime.0.chromosome.txt')
            parse_runtime_file('StrainGST', 'all', '1', 'straingst_runtime.1.chromosome.txt')
            parse_runtime_file('StrainGST', 'all', '2', 'straingst_runtime.2.chromosome.txt')
            parse_runtime_file('StrainGST', 'all', '3', 'straingst_runtime.3.chromosome.txt')
            parse_runtime_file('StrainGST', 'all', '4', 'straingst_runtime.4.chromosome.txt')
            parse_runtime_file('StrainFacts', 'GTPro', 'all', 'gtpro_runtime.txt')
            parse_runtime_file('StrainFacts', 'Inference', 'all', 'strainfacts_runtime.txt')
            parse_runtime_file('StrainEst (Sensitive)', 'Inference', '0', 'strainest_runtime.sensitive.0.txt')
            parse_runtime_file('StrainEst (Sensitive)', 'Inference', '1', 'strainest_runtime.sensitive.1.txt')
            parse_runtime_file('StrainEst (Sensitive)', 'Inference', '2', 'strainest_runtime.sensitive.2.txt')
            parse_runtime_file('StrainEst (Sensitive)', 'Inference', '3', 'strainest_runtime.sensitive.3.txt')
            parse_runtime_file('StrainEst (Sensitive)', 'Inference', '4', 'strainest_runtime.sensitive.4.txt')
            parse_runtime_file('StrainEst (Default)', 'Inference', '0', 'strainest_runtime.default.0.txt')
            parse_runtime_file('StrainEst (Default)', 'Inference', '1', 'strainest_runtime.default.1.txt')
            parse_runtime_file('StrainEst (Default)', 'Inference', '2', 'strainest_runtime.default.2.txt')
            parse_runtime_file('StrainEst (Default)', 'Inference', '3', 'strainest_runtime.default.3.txt')
            parse_runtime_file('StrainEst (Default)', 'Inference', '4', 'strainest_runtime.default.4.txt')
    return pd.DataFrame(df_entries)


def main():
    args = parse_args()
    result_base_dir = Path(args.base_data_dir)
    out_dir = Path(args.out_dir)

    # Necessary precomputation.
    ground_truth = load_ground_truth(Path(args.ground_truth_path))
    index_df = pd.read_csv(args.index_path, sep='\t')
    chronostrain_db = cfg.database_cfg.get_database()
    out_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Evaluating error metrics.")
    summary_df = evaluate_errors(
        ground_truth,
        result_base_dir,
        chronostrain_db
    )
    out_path = out_dir / 'summary.csv'
    summary_df.to_csv(out_path, index=False)
    logger.info(f"[*] Saved error metrics to {out_path}.")

    logger.info("Evaluating runtimes.")
    runtime_df = evaluate_runtimes(result_base_dir)
    out_path = out_dir / 'runtime.csv'
    runtime_df.to_csv(out_path, index=False)
    logger.info(f"[*] Saved runtime summaries to {out_path}.")


if __name__ == "__main__":
    main()
