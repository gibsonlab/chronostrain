from typing import *
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special
import sklearn.metrics

from chronostrain.inference import GaussianWithGlobalZerosPosteriorDense, GaussianWithGumbelsPosterior
from chronostrain.config import cfg

from .base import trial_dir, parse_runtime, parse_phylogroups
from .ground_truth import load_ground_truth
from .error_metrics import rms, tv_error, compute_rank_corr, binned_rms, strain_split_rms


def extract_chronostrain_prediction(
        mut_ratio: str, replicate: int, read_depth: int, trial: int, subdir_name: str = 'chronostrain'
) -> Tuple[GaussianWithGlobalZerosPosteriorDense, List[str], Dict, int]:
    target_accs, time_points, ground_truth = load_ground_truth(mut_ratio=mut_ratio, replicate=replicate)
    output_basedir = trial_dir(mut_ratio, replicate, read_depth, trial) / 'output'
    output_dir = output_basedir / subdir_name
    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output for replicate={replicate}, read_depth={read_depth}, trial={trial} does not exist.")

    # samples = np.load(str(output_dir / 'samples.npy'))
    strains = parse_chronostrain_strains(output_dir / 'strains.txt')

    posterior = GaussianWithGlobalZerosPosteriorDense(len(strains), len(time_points), cfg.engine_cfg.dtype)
    posterior.load(Path(output_dir / "posterior.{}.npz".format(cfg.engine_cfg.dtype)))

    adhoc_clusters = parse_adhoc_clusters(output_dir / 'adhoc_cluster.txt')
    runtime = parse_runtime(output_dir / 'filter_runtime.txt') + parse_runtime(output_dir / 'inference_runtime.txt')
    return posterior, strains, adhoc_clusters, runtime


def parse_chronostrain_strains(txt_file) -> List[str]:
    with open(txt_file, 'rt') as f:
        return [x.strip() for x in f]


def parse_adhoc_clusters(txt_file) -> Dict[str, str]:
    clust = {}
    with open(txt_file, "rt") as f:
        for line in f:
            tokens = line.strip().split(":")
            rep = tokens[0]
            members = tokens[1].split(",")
            for member in members:
                clust[member] = rep
    return clust


def load_chronostrain_cluster(chronostrain_cluster: Path) -> pd.DataFrame:
    df_entries = []
    phylogroups = parse_phylogroups()
    with open(chronostrain_cluster, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            tokens = line.strip().split("\t")
            rep_id = tokens[0]
            members = tokens[1].split(",")
            for member in members:
                df_entries.append({
                    'Accession': member,
                    'Cluster': rep_id,
                    'ClusterPhylogroup': phylogroups[rep_id]
                })
    return pd.DataFrame(df_entries)


def extract_prediction_subset_bf_conditional(
        truth_accs: List[str],
        ground_truth: np.ndarray,
        posterior: GaussianWithGumbelsPosterior,
        g_samples: np.ndarray,
        z_samples: np.ndarray,
        prior_p: float,
        clust_subset: List[str],
        strain_to_idx: Dict[str, int],
        chronostrain_clustering_df: pd.DataFrame,
        posterior_threshold: float
):
    """
    First, compute samples from q(Abundance | Z=z), where (z) is the indicator vector that filters only strain clusters with BF > bf_threshold.
    Then, before returning, restrict the view to [target_clusters] and re-normalize.

    Note: this is NOT the same as q(Abundance | Z=target_clusters), which samples trajectories conditional on ALL strains in the target being present, regardless of bayes factors.
    """
    n_timepoints = ground_truth.shape[0]
    n_samples = g_samples.shape[1]

    clust_indices = {c: i for i, c in enumerate(clust_subset)}
    SUBSET_pred = np.zeros((n_timepoints, n_samples, len(clust_subset)), dtype=float)
    SUBSET_truth = np.zeros((n_timepoints, len(clust_subset)), dtype=float)
    SUBSET_pred_probs = np.zeros(len(clust_subset), dtype=float)
    SUBSET_true_labels = np.full(len(clust_subset), dtype=bool, fill_value=False)

    # Calculate bayes factors.
    posterior_inclusion_p = scipy.special.expit(-posterior.get_parameters()['gumbel_diff'])
    posterior_inclusion_bf = (posterior_inclusion_p / (1 - posterior_inclusion_p)) * ((1 - prior_p) / prior_p)

    # Calculate abundance estimates using BF thresholds.
    indicators = np.full(len(strain_to_idx), fill_value=False, dtype=bool)
    # indicators[posterior_inclusion_bf > 100000.0] = True
    indicators[posterior_inclusion_p > posterior_threshold] = True
    log_indicators = np.empty(len(strain_to_idx), dtype=float)
    log_indicators[indicators] = 0.0
    log_indicators[~indicators] = -np.inf
    pred_abundances = scipy.special.softmax(g_samples + np.expand_dims(log_indicators, axis=[0, 1]), axis=-1)

    # Fill in the prediction matrices. (Undo the ad-hoc clustering by dividing the abundance.)
    for tgt_idx, tgt_clust in enumerate(clust_subset):
        adhoc_clust_id = chronostrain_clustering_df.loc[
            chronostrain_clustering_df['Cluster'] == tgt_clust, 'Adhoc'].head(1).item()  # adhoc_clusters[tgt_clust]
        adhoc_idx = strain_to_idx[adhoc_clust_id]
        adhoc_clust_sz = len(
            pd.unique(chronostrain_clustering_df.loc[chronostrain_clustering_df['Adhoc'] == adhoc_clust_id, 'Cluster'])
        )

        # divide by the adhoc clust size.
        SUBSET_pred[:, :, tgt_idx] = (pred_abundances[:, :, adhoc_idx] / adhoc_clust_sz) * int(indicators[adhoc_idx])
        SUBSET_pred_probs[tgt_idx] = posterior_inclusion_p[adhoc_idx]

    # Create the matching ground-truth clustered abundance.
    for tgt_idx, tgt_acc in enumerate(truth_accs):
        cluster_acc = chronostrain_clustering_df.loc[
            chronostrain_clustering_df['Accession'] == tgt_acc,
            'Cluster'
        ].item()
        clust_idx = clust_indices[cluster_acc]
        SUBSET_truth[:, clust_idx] = ground_truth[:, tgt_idx]
        SUBSET_true_labels[clust_idx] = True

    SUBSET_pred = SUBSET_pred / SUBSET_pred.sum(axis=-1, keepdims=True)
    return SUBSET_pred, SUBSET_truth, SUBSET_pred_probs, SUBSET_true_labels


def load_chronostrain_posterior_samples(posterior: GaussianWithGlobalZerosPosteriorDense, n_samples: int, save_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    sample_path = save_dir / 'samples.npz'
    if not sample_path.exists():
        rand = posterior.random_sample(n_samples)
        g_samples = np.array(
            posterior.reparametrized_gaussians(rand['std_gaussians'], posterior.get_parameters()))  # T x N x S
        z_samples = np.array(posterior.reparametrized_zeros(rand['std_gumbels'], posterior.get_parameters()))  # N x S
        np.savez(sample_path, g=g_samples, z=z_samples)
    else:
        samples = np.load(sample_path)
        g_samples = samples['g']
        z_samples = samples['z']
    return g_samples, z_samples


def chronostrain_results(
    mut_ratio: str,
    replicate: int, 
    read_depth: int, 
    trial: int,
    abundance_bins: np.ndarray,
    subdir_name: str = 'chronostrain', 
    posterior_threshold: float = 0.9901
):
    truth_accs, time_points, ground_truth = load_ground_truth(mut_ratio=mut_ratio, replicate=replicate)
    posterior, strains, adhoc_clusters, runtime = extract_chronostrain_prediction(mut_ratio, replicate, read_depth, trial, subdir_name=subdir_name)

    strain_to_idx = {s: i for i, s in enumerate(strains)}
    chronostrain_clustering_df = load_chronostrain_cluster(Path('/mnt/e/ecoli_db/chronostrain_files/ecoli.clusters.txt'))
    chronostrain_clustering_df = chronostrain_clustering_df.assign(
        Adhoc=chronostrain_clustering_df['Cluster'].map(adhoc_clusters)
    )

    """ Extract phylogroup A prediction/truth. """
    # First, check that the four target genomes are in distinct clusters.
    sim_clusts = chronostrain_clustering_df.loc[
        chronostrain_clustering_df['Accession'].isin(set(truth_accs)), 'Cluster']
    if len(truth_accs) != len(pd.unique(sim_clusts)):
        raise ValueError(
            "Resulting clustering does not separate the {} target accessions for replicate {}, read_depth {}, trial {}".format(
                len(truth_accs), replicate, read_depth, trial))

    # Extract samples.
    n_samples = 5000
    sample_dir = Path() / f'{subdir_name}_samples' / f'mutratio_{mut_ratio}' / f'replicate_{replicate}' / f'reads_{read_depth}' / f'trial_{trial}'  # relative dir
    sample_dir.mkdir(exist_ok=True, parents=True)
    g_samples, z_samples = load_chronostrain_posterior_samples(posterior, n_samples, save_dir=sample_dir)

    """ Extract phylogroup A prediction. """
    phylogroup_A_clusts = list(pd.unique(
        chronostrain_clustering_df.loc[chronostrain_clustering_df['ClusterPhylogroup'] == 'A', 'Cluster']
    ))
    A_pred, A_truth, A_pred_indicators, A_true_indicators = extract_prediction_subset_bf_conditional(
        truth_accs,
        ground_truth,
        posterior,
        g_samples, z_samples, 0.001,
        phylogroup_A_clusts,
        strain_to_idx,
        chronostrain_clustering_df,
        posterior_threshold=posterior_threshold
    )
    # A_pred, A_truth, A_pred_indicators, A_true_indicators = extract_prediction_subset_full(posterior, phylogroup_A_clusts, strain_to_idx, chronostrain_clustering_df)

    """ Extract sim-only prediction. """
    sim_clusts = list(pd.unique(sim_clusts))
    sim_pred, sim_truth, sim_pred_indicators, sim_true_indicators = extract_prediction_subset_bf_conditional(
        truth_accs,
        ground_truth,
        posterior,
        g_samples, z_samples, 0.001,
        sim_clusts,
        strain_to_idx,
        chronostrain_clustering_df,
        posterior_threshold=posterior_threshold
    )  # condition on all strains being present, renormalize. (Essentially just measures the gaussian part)

    # ================ Metric evaluation
    # Median trajectories.
    A_median_traj = np.median(A_pred, axis=1)
    A_median_traj = A_median_traj / np.sum(A_median_traj, axis=-1, keepdims=True)  # renormalize
    sim_median_traj = np.median(sim_pred, axis=1)
    sim_median_traj = sim_median_traj / np.sum(sim_median_traj, axis=-1, keepdims=True)

    auroc_abund = sklearn.metrics.roc_auc_score(  # Abundance thresholding
        y_true=np.tile(A_true_indicators, (len(time_points), 1)).flatten(),
        y_score=A_median_traj.flatten(),
    )
    auroc_posterior = sklearn.metrics.roc_auc_score(  # posterior score
        y_true=A_true_indicators,
        y_score=A_pred_indicators
    )

    tv_err_sim = tv_error(sim_median_traj, sim_truth)
    tv_err_A = tv_error(A_median_traj, A_truth)

    # tv_err_A = np.median(tv_error(A_pred, A_truth))
    # tv_err_sim = np.median(tv_error(sim_pred, sim_truth))

    eps = 1e-4
    # rms_error_sim = np.median(rms(np.log10(sim_pred + eps), np.log10(sim_truth + eps)))
    # rms_error_A = np.median(rms(np.log10(A_pred + eps), np.log10(A_truth + eps)))
    rms_error_sim = rms(np.log10(sim_median_traj + eps), np.log10(sim_truth))
    rms_error_A = rms(np.log10(A_median_traj + eps), np.log10(A_truth + eps))

    rank_corr_sim = np.median(compute_rank_corr(sim_median_traj, sim_truth))
    rank_corr_A = np.median(compute_rank_corr(A_median_traj, A_truth))

    binned_rms_error_sim = binned_rms(np.log10(sim_median_traj + eps), np.log10(sim_truth), abundance_bins)
    split_rms_sim = strain_split_rms(np.log10(sim_median_traj + eps), np.log10(sim_truth))

    # ================= Output
    return {
        'MutRatio': mut_ratio, 'Replicate': replicate, 'ReadDepth': read_depth, 'Trial': trial,
        'Method': 'ChronoStrain',
        'NumClusters': len(phylogroup_A_clusts),
        'TVErrorSim': tv_err_sim,
        'TVErrorA': tv_err_A,
        'RMSErrorSim': rms_error_sim,
        'RMSErrorA': rms_error_A,
        'AUROC_abundance': auroc_abund,
        'AUROC_posterior': auroc_posterior,
        'RankCorrelationSim': rank_corr_sim,
        'RankCorrelationA': rank_corr_A,
        'Runtime': runtime
    } | {
        f'RMSErrorSim_Bin{i}': binned_rms
        for i, binned_rms in enumerate(binned_rms_error_sim)
    } | {
        f'RMSErrorSim_Strain{i}': _rms
        for i, _rms in enumerate(split_rms_sim)
    }


def chronostrain_roc(mut_ratio: str, replicate: int, read_depth: int, trial: int) -> Tuple[np.ndarray, np.ndarray]:
    truth_accs, time_points, ground_truth = load_ground_truth(mut_ratio=mut_ratio, replicate=replicate)
    posterior, strains, adhoc_clusters, runtime = extract_chronostrain_prediction(mut_ratio, replicate, read_depth, trial)

    strain_to_idx = {s: i for i, s in enumerate(strains)}
    chronostrain_clustering_df = load_chronostrain_cluster(Path('/mnt/e/ecoli_db/chronostrain_files/ecoli.clusters.txt'))
    chronostrain_clustering_df = chronostrain_clustering_df.assign(
        Adhoc=chronostrain_clustering_df['Cluster'].map(adhoc_clusters)
    )

    """ Extract phylogroup A prediction/truth. """
    # First, check that the four target genomes are in distinct clusters.
    sim_clusts = chronostrain_clustering_df.loc[
        chronostrain_clustering_df['Accession'].isin(set(truth_accs)), 'Cluster']
    if len(truth_accs) != len(pd.unique(sim_clusts)):
        raise ValueError(
            "Resulting clustering does not separate the {} target accessions for replicate {}, read_depth {}, trial {}".format(
                len(truth_accs), replicate, read_depth, trial))

    # Extract samples.
    n_samples = 5000
    sample_dir = Path() / 'chronostrain_samples' / f'mutratio_{mut_ratio}' / f'replicate_{replicate}' / f'reads_{read_depth}' / f'trial_{trial}'  # relative dir
    sample_dir.mkdir(exist_ok=True, parents=True)
    g_samples, z_samples = load_chronostrain_posterior_samples(posterior, n_samples, save_dir=sample_dir)

    """ Extract phylogroup A prediction. """
    phylogroup_A_clusts = list(pd.unique(
        chronostrain_clustering_df.loc[chronostrain_clustering_df['ClusterPhylogroup'] == 'A', 'Cluster']
    ))
    A_pred, A_truth, A_pred_indicators, A_true_indicators = extract_prediction_subset_bf_conditional(
        truth_accs,
        ground_truth,
        posterior,
        g_samples, z_samples, 0.001,
        phylogroup_A_clusts,
        strain_to_idx,
        chronostrain_clustering_df,
        posterior_threshold=0.9901
    )

    # ================ Metric evaluation
    # Median trajectories.
    A_median_traj = np.median(A_pred, axis=1)
    A_median_traj = A_median_traj / np.sum(A_median_traj, axis=-1, keepdims=True)  # renormalize

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(  # Abundance thresholding
        y_true=np.tile(A_true_indicators, (len(time_points), 1)).flatten(),
        y_score=A_median_traj.flatten(),
    )
    return fpr, tpr
