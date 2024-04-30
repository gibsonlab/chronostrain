from typing import *
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
from collections import defaultdict

import numpy as np
import sklearn.metrics

from .base import trial_dir, parse_runtime, parse_phylogroups
from .ground_truth import load_ground_truth
from .error_metrics import rms, tv_error, compute_rank_corr, binned_rms, strain_split_rms
from .mgems import parse_msweep_pred_single, msweep_subset_prediction


def load_demix_check(demix_check_path: Path, cluster_order: List[str]) -> np.ndarray:
    if not demix_check_path.exists():
        raise FileNotFoundError("demix_check not done at {}".format(demix_check_path))
    
    # '5' is a fictitious score, even worse than '4' (demix_check didn't run for this cluster). This is a tuple; second entry is the number of binned reads.
    demix_check_scores = defaultdict(lambda: (5, 0))
    try:
        demix_check_df = pd.read_csv(demix_check_path, sep='\t')
    except EmptyDataError:
        raise DemixCheckException(f"demix_check output was empty ({demix_check_path}).")
        
    if 'cluster' not in demix_check_df.columns:
        raise DemixCheckException("Malformatted demix_check output {}. Found DF indices {}".format(demix_check_path, demix_check_df.columns))
    if demix_check_df.shape[0] == 0:
        raise DemixCheckException(f"demix_check output was empty ({demix_check_path}).")
    for _, row in demix_check_df.iterrows():
        clust_id = str(row['cluster'])
        clust_score = int(row['score'])
        if pd.isna(row['read_count']):
            n_reads = 0
            clust_score = 5  # no binned reads; downgrade the score (demix_check shouldn't have been computed for these)
        else:
            n_reads = int(row['read_count'])
        demix_check_scores[clust_id] = (clust_score, n_reads)
    return np.array([demix_check_scores[c] for c in cluster_order], dtype=int)


def extract_msweep_hierarchical_prediction(
        mut_ratio: str,
        replicate: int,
        read_depth: int,
        trial: int,
        time_points: List[float],
        subdir_name: str = "mgems"
) -> Tuple[np.ndarray, List[str], pd.DataFrame, int]:
    cluster_df = load_msweep_ecoli_cluster()
    cluster_ordering = sorted(pd.unique(cluster_df['Cluster']))

    output_basedir = trial_dir(mut_ratio, replicate, read_depth, trial) / 'output'
    output_basedir_cctm = trial_dir(mut_ratio, replicate, read_depth, trial, base_dir=Path('/data/cctm/youn/chronostrain_semisynthetic')) / 'output'
    
    output_dir = output_basedir_cctm / subdir_name
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir for mut_ratio={mut_ratio}, replicate={replicate}, read_depth={read_depth}, trial={trial} not found.")

    preds = np.zeros((len(time_points), len(cluster_ordering)), dtype=float)
    for t_idx in range(len(time_points)):
        t_path = output_dir / f'{t_idx}' / 'Ecoli' / 'msweep_abundances.txt'
        if t_path.exists():
            preds[t_idx, :] = parse_msweep_pred_single(t_path, cluster_ordering)
        else:
            raise FileNotFoundError(f"mSWEEP output for t_idx={t_idx} not found. (path={t_path})")

    # renormalize.
    preds = preds / preds.sum(axis=-1, keepdims=True)
    # demix_check_scores = load_demix_check(output_dir / f'{t_idx}' / 'Ecoli' / 'demix_check' / 'clu_score.tsv', cluster_ordering)

    runtime = 0
    try:
        for i in range(len(time_points)):
            runtime += parse_runtime(output_basedir / subdir_name / f'{t_idx}' / 'Ecoli' / f'runtime.{t_idx}.txt')
            runtime += parse_runtime(output_basedir / subdir_name / f'{t_idx}' / 'species' / f'runtime.{t_idx}.txt')
    except FileNotFoundError:
        runtime = np.nan
        
    return preds, cluster_ordering, cluster_df, runtime


def load_msweep_ecoli_cluster() -> pd.DataFrame:
    msweep_clust_df = pd.read_csv("/mnt/e/infant_nt/database/mgems/ref_dir/Ecoli/ref_clu.tsv", sep='\t')
    msweep_clust_df = msweep_clust_df.rename(columns={'id': 'Accession', 'cluster': 'Cluster'})
    msweep_clust_df['Cluster'] = msweep_clust_df['Cluster'].astype(str)

    phylogroups = parse_phylogroups()
    msweep_clust_df['MemberPhylogroup'] = msweep_clust_df['Accession'].map(phylogroups)
    msweep_clust_df['MemberPhylogroupA'] = msweep_clust_df['MemberPhylogroup'] == 'A'

    # Note: here, a cluster is designated "phylogroup A" if at least half its members is phylogroup A.
    phylo_A_count = msweep_clust_df.groupby('Cluster')['MemberPhylogroupA'].sum().rename('Phylogroup_A_Count')
    group_sizes = msweep_clust_df.groupby('Cluster')['Accession'].count().rename('GroupSize')
    A_ratios = (phylo_A_count / group_sizes).rename("A_Ratio").reset_index()

    # This dataframe keeps track of the ratio of phylogroup A to the total group size.
    msweep_clust_df = msweep_clust_df.merge(A_ratios, on='Cluster')
    return msweep_clust_df


def msweep_hierarchical_results(
    mut_ratio: str, 
    replicate: int, 
    read_depth: int, 
    trial: int, 
    abundance_bins: np.ndarray,
    subdir_name: str = 'mgems',
    lod: float = 0.002
):
    target_accs, time_points, ground_truth = load_ground_truth(mut_ratio=mut_ratio, replicate=replicate)
    pred, clusters, msweep_clust_df, runtime = extract_msweep_hierarchical_prediction(
        mut_ratio, replicate, read_depth, trial, time_points, subdir_name=subdir_name
    )
    msweep_ordering = {c: i for i, c in enumerate(clusters)}
    return compile_into_results(
        mut_ratio, replicate, read_depth, trial, abundance_bins, lod,
        target_accs, time_points, ground_truth, pred, clusters, msweep_clust_df, runtime, msweep_ordering
    )


def compile_into_results(
    mut_ratio, replicate, read_depth, trial, abundance_bins, lod,
    target_accs, time_points, ground_truth, pred, clusters, msweep_clust_df, runtime, msweep_ordering
):
    """ Extract phylogroup A prediction/truth. """
    # First, check that the four target genomes are in distinct clusters.
    sim_clusts = msweep_clust_df.loc[msweep_clust_df['Accession'].isin(set(target_accs)), 'Cluster']
    if len(target_accs) != len(pd.unique(sim_clusts)):
        raise ValueError(
            "Resulting clustering does not separate the {} target accessions for replicate {}, read_depth {}, trial {}".format(
                len(target_accs), replicate, read_depth, trial))

    # Next initialize the ground truth/prediction matrices.
    A_clusts = list(pd.unique(msweep_clust_df.loc[msweep_clust_df['A_Ratio'] > 0.5, 'Cluster']))
    A_pred, A_truth, A_true_indicators = msweep_subset_prediction(target_accs, ground_truth, pred, msweep_ordering, A_clusts, msweep_clust_df)

    # Simulated strains only
    sim_clusts = list(pd.unique(sim_clusts))
    sim_pred, sim_truth, sim_true_indicators = msweep_subset_prediction(target_accs, ground_truth, pred, msweep_ordering, sim_clusts, msweep_clust_df)

    # Note: at this point, A_pred and sim_pred are the sub-matrices of the RAW mSWEEP output predictions. No re-normalizations done.
    # ================ Metric evaluation ================
    auroc = sklearn.metrics.roc_auc_score(  # Abundance thresholding per timepoint
        y_true=np.tile(A_true_indicators, (len(time_points), 1)).flatten(),
        y_score=A_pred.flatten(),
    )
    auroc_collapsed = sklearn.metrics.roc_auc_score(  # Abundance thresholding, collapsed
        y_true=A_true_indicators,
        y_score=np.max(A_pred, axis=0),
    )
    
    A_pred_filtered_renorm = np.copy(A_pred)
    A_pred_filtered_renorm[A_pred_filtered_renorm < lod] = 0.0
    A_pred_filtered_renorm = A_pred_filtered_renorm / np.sum(A_pred_filtered_renorm, axis=-1, keepdims=True)
    
    sim_pred_filtered_renorm = np.copy(sim_pred)
    sim_pred_filtered_renorm[sim_pred_filtered_renorm < lod] = 0.0
    sim_pred_filtered_renorm = sim_pred_filtered_renorm / np.sum(sim_pred_filtered_renorm, axis=-1, keepdims=True)
    
    #### ===== Question: compute rank corrs BEFORE of AFTER thresholding zeros?
    rank_corr_sim = compute_rank_corr(sim_pred_filtered_renorm, sim_truth)
    rank_corr_A = compute_rank_corr(A_pred_filtered_renorm, A_truth)

    eps = 1e-4
    rms_error_sim = rms(np.log10(sim_pred_filtered_renorm + eps), np.log10(sim_truth + eps))
    rms_error_A = rms(np.log10(A_pred_filtered_renorm + eps), np.log10(A_truth + eps))

    tv_err_A = tv_error(A_pred_filtered_renorm, A_truth)
    tv_err_sim = tv_error(sim_pred_filtered_renorm, sim_truth)

    binned_rms_error_sim = binned_rms(np.log10(sim_pred_filtered_renorm + eps), np.log10(sim_truth), abundance_bins)
    split_rms_sim = strain_split_rms(np.log10(sim_pred_filtered_renorm + eps), np.log10(sim_truth))

    # ================= Output
    return {
        'MutRatio': mut_ratio, 'Replicate': replicate, 'ReadDepth': read_depth, 'Trial': trial,
        'Method': 'mGEMS-h',
        'NumClusters': len(A_clusts),
        'TVErrorSim': tv_err_sim,
        'TVErrorA': tv_err_A,
        'RMSErrorSim': rms_error_sim,
        'RMSErrorA': rms_error_A,
        'AUROC': auroc,
        'AUROC_Collapsed': auroc_collapsed,
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


def msweep_hierarchical_roc(
    mut_ratio: str, 
    replicate: int, 
    read_depth: int, 
    trial: int, 
    subdir_name: str = 'mgems',
):
    target_accs, time_points, ground_truth = load_ground_truth(mut_ratio=mut_ratio, replicate=replicate)
    pred, clusters, msweep_clust_df, runtime = extract_msweep_hierarchical_prediction(
        mut_ratio, replicate, read_depth, trial, time_points, subdir_name=subdir_name
    )
    msweep_ordering = {c: i for i, c in enumerate(clusters)}

    """ Extract phylogroup A prediction/truth. """
    # First, check that the four target genomes are in distinct clusters.
    sim_clusts = msweep_clust_df.loc[msweep_clust_df['Accession'].isin(set(target_accs)), 'Cluster']
    if len(target_accs) != len(pd.unique(sim_clusts)):
        raise ValueError(
            "Resulting clustering does not separate the {} target accessions for replicate {}, read_depth {}, trial {}".format(
                len(target_accs), replicate, read_depth, trial))

    # Next initialize the ground truth/prediction matrices.
    A_clusts = list(pd.unique(msweep_clust_df.loc[msweep_clust_df['A_Ratio'] > 0.5, 'Cluster']))
    A_pred, A_truth, A_true_indicators = msweep_subset_prediction(target_accs, ground_truth, pred, msweep_ordering, A_clusts, msweep_clust_df)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(  # Abundance thresholding, (TxS)
        y_true=np.tile(A_true_indicators, (len(time_points), 1)).flatten(),
        y_score=A_pred.flatten(),
    )
    return fpr, tpr
