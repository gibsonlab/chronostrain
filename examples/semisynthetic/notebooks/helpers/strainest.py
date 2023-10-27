from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .base import trial_dir, parse_runtime, parse_phylogroups
from .ground_truth import load_ground_truth
from .error_metrics import rms, tv_error, compute_rank_corr


class StrainEstInferenceError(BaseException):
    pass


def extract_strainest_prediction(
        mut_ratio: str,
        replicate: int,
        read_depth: int,
        trial: int,
        time_points: List[float]
) -> Tuple[np.ndarray, List[str], pd.DataFrame, int]:
    output_basedir = trial_dir(mut_ratio, replicate, read_depth, trial) / 'output'

    cluster_df = load_strainest_cluster()  # we ran strainest with poppunk clusters, same as in msweep.
    cluster_ordering = sorted(pd.unique(cluster_df['Cluster']))

    output_dir = output_basedir / 'strainest'
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir for mut_ratio={mut_ratio}, replicate={replicate}, read_depth={read_depth}, trial={trial} not found.")

    preds = np.zeros((len(time_points), len(cluster_ordering)), dtype=float)
    for t_idx in range(len(time_points)):
        t_path = output_dir / f'abund_{t_idx}.txt'
        if t_path.exists():
            pred_t = parse_strainest_pred_single(t_path, cluster_ordering)
            preds[t_idx, :] = pred_t
        else:
            raise FileNotFoundError(f"StrainEst output for t_idx={t_idx} not found. (path={t_path})")

    # renormalize.
    preds = preds / preds.sum(axis=-1, keepdims=True)

    runtime = 0
    for t_idx in range(len(time_points)):
        runtime += parse_runtime(output_basedir / f'strainest_runtime.{t_idx}.txt')
    return preds, cluster_ordering, cluster_df, runtime


def parse_strainest_pred_single(pred_file: Path, cluster_ordering: List[str]) -> np.ndarray:
    abund_t = np.zeros(len(cluster_ordering), dtype=float)
    clust_to_idx = {s: i for i, s in enumerate(cluster_ordering)}

    df = pd.read_csv(pred_file, sep='\t')
    columns = df.columns
    abund_col = columns[1]
    for _, row in df.iterrows():
        acc = row['OTU']
        if acc.endswith('.chrom.fna'):
            acc = acc[:-len('.chrom.fna')]
        abund = float(row[abund_col])
        abund_t[clust_to_idx[acc]] = abund
    if np.sum(abund_t) == 0:
        raise StrainEstInferenceError()
    return abund_t


def load_strainest_cluster() -> pd.DataFrame:
    p = Path("/mnt/e/semisynthetic_data/databases/StrainEst/snv_profiling_db/cluster_members.txt")
    df = pd.read_csv(p, sep='\t')

    phylogroups = parse_phylogroups()
    df['ClusterPhylogroup'] = df['Cluster'].map(phylogroups)
    return df


def strainest_subset_prediction(
        truth_accs: List[str],
        ground_truth: np.ndarray,
        pred: np.ndarray,
        pred_ordering: Dict[str, int],
        clust_subset: List[str],
        strainest_clust_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_timepoints = pred.shape[0]
    clust_indices = {c: i for i, c in enumerate(clust_subset)}  # phylogroup A only
    SUBSET_pred = np.zeros((n_timepoints, len(clust_subset)), dtype=float)
    SUBSET_truth = np.zeros((n_timepoints, len(clust_subset)), dtype=float)
    SUBSET_true_labels = np.full(len(clust_subset), dtype=bool, fill_value=False)

    # Populate the abundances.
    for tgt_idx, tgt_clust in enumerate(clust_subset):
        pred_idx = pred_ordering[tgt_clust]
        SUBSET_pred[:, tgt_idx] = pred[:, pred_idx]

    # Create the matching ground-truth clustered abundance.
    for sim_index, sim_acc in enumerate(truth_accs):
        cluster_id = strainest_clust_df.loc[strainest_clust_df['Accession'] == sim_acc, 'Cluster'].item()
        tgt_idx = clust_indices[cluster_id]  # this will throw an error if the cluster is not phylogroup A
        SUBSET_truth[:, tgt_idx] = ground_truth[:, sim_index]
        SUBSET_true_labels[tgt_idx] = True
    if np.equal(np.sum(SUBSET_pred, axis=-1), 0).sum() > 0:
        raise StrainEstInferenceError()
    SUBSET_pred = SUBSET_pred / np.sum(SUBSET_pred, axis=-1, keepdims=True)
    return SUBSET_pred, SUBSET_truth, SUBSET_true_labels


def strainest_results(mut_ratio: str, replicate: int, read_depth: int, trial: int):
    truth_accs, time_points, ground_truth = load_ground_truth(replicate=replicate)
    pred, clusters, strainest_clust_df, runtime = extract_strainest_prediction(
        mut_ratio, replicate, read_depth, trial, time_points
    )
    strainest_ordering = {c: i for i, c in enumerate(clusters)}

    """ Extract phylogroup A prediction/truth. """
    # First, check that the four target genomes are in distinct clusters.
    sim_clusts = strainest_clust_df.loc[strainest_clust_df['Accession'].isin(set(truth_accs)), 'Cluster']
    if len(truth_accs) != len(pd.unique(sim_clusts)):
        raise ValueError(
            "Resulting clustering does not separate the {} target accessions for replicate {}, read_depth {}, trial {}".format(
                len(truth_accs), replicate, read_depth, trial))

    # Next initialize the ground truth/prediction matrices.
    phylogroup_A_clusts = list(pd.unique(strainest_clust_df.loc[strainest_clust_df['ClusterPhylogroup'] == 'A', 'Cluster']))
    A_pred, A_truth, A_true_indicators = strainest_subset_prediction(
        truth_accs, ground_truth,
        pred, strainest_ordering,
        phylogroup_A_clusts, strainest_clust_df
    )

    # Simulated strains only
    sim_clusts = list(pd.unique(sim_clusts))
    sim_pred, sim_truth, sim_true_indicators = strainest_subset_prediction(
        truth_accs, ground_truth,
        pred, strainest_ordering,
        sim_clusts, strainest_clust_df
    )

    # ================ Metric evaluation
    auroc = roc_auc_score(  # Abundance thresholding per timepoint
        y_true=np.tile(A_true_indicators, (len(time_points), 1)).flatten(),
        y_score=A_pred.flatten()
    )

    eps = 1e-6
    rms_error_sim = rms(np.log10(sim_pred + eps), np.log10(sim_truth + eps))
    rms_error_A = rms(np.log10(A_pred + eps), np.log10(A_truth + eps))

    tv_err_A = tv_error(A_pred, A_truth)
    tv_err_sim = tv_error(sim_pred, sim_truth)

    rank_corr_sim = np.median(compute_rank_corr(sim_pred, sim_truth))
    rank_corr_A = np.median(compute_rank_corr(A_pred, A_truth))

    # ================= Output
    return {
        'MutRatio': mut_ratio, 'Replicate': replicate, 'ReadDepth': read_depth, 'Trial': trial,
        'Method': 'StrainEst',
        'NumClusters': len(phylogroup_A_clusts),
        'TVErrorSim': tv_err_sim,
        'TVErrorA': tv_err_A,
        'RMSErrorSim': rms_error_sim,
        'RMSErrorA': rms_error_A,
        'AUROC': auroc,
        'RankCorrelationSim': rank_corr_sim,
        'RankCorrelationA': rank_corr_A,
        'Runtime': runtime
    }
