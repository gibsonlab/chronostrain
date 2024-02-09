from typing import *
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn.metrics

from .base import trial_dir, parse_runtime, parse_phylogroups
from .ground_truth import load_ground_truth
from .error_metrics import rms, tv_error, compute_rank_corr, binned_rms, strain_split_rms


def extract_msweep_prediction(
        mut_ratio: str,
        replicate: int,
        read_depth: int,
        trial: int,
        time_points: List[float]
) -> Tuple[np.ndarray, List[str], pd.DataFrame, int]:
    cluster_df = load_msweep_cluster()
    cluster_ordering = sorted(pd.unique(cluster_df['Cluster']))

    output_basedir = trial_dir(mut_ratio, replicate, read_depth, trial) / 'output'
    output_dir = output_basedir / 'msweep'
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir for mut_ratio={mut_ratio}, replicate={replicate}, read_depth={read_depth}, trial={trial} not found.")

    preds = np.zeros((len(time_points), len(cluster_ordering)), dtype=float)
    for t_idx in range(len(time_points)):
        t_path = output_dir / f'{t_idx}_abundances.txt'
        if t_path.exists():
            preds[t_idx, :] = parse_msweep_pred_single(t_path, cluster_ordering)
        else:
            raise FileNotFoundError(f"mSWEEP output for t_idx={t_idx} not found. (path={t_path})")

    # renormalize.
    preds = preds / preds.sum(axis=-1, keepdims=True)

    runtimes = [parse_runtime(output_basedir / 'themisto' / 'runtime.txt')]
    runtimes += [
        parse_runtime(output_basedir / 'msweep' / f'runtime.{i}.txt')
        for i in range(len(time_points))
    ]
    runtime = np.sum(runtimes)
    return preds, cluster_ordering, cluster_df, runtime


def parse_msweep_pred_single(target_path: Path, cluster_ordering: List[str]) -> np.ndarray:
    result = np.zeros(len(cluster_ordering), dtype=float)
    clust_to_idx = {s: i for i, s in enumerate(cluster_ordering)}
    clusters_remaining = set(cluster_ordering)
    with open(target_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip()
            if len(line) == 0:
                continue

            clust, abund_str = line.split('\t')
            if clust not in clust_to_idx:
                continue
            result[clust_to_idx[clust]] = float(abund_str)
            clusters_remaining.remove(clust)
    if len(clusters_remaining) > 0:
        print("[WARNING]: accessions {} not found in {}".format(clusters_remaining, target_path))
    return result


def load_msweep_cluster() -> pd.DataFrame:
    db_dir = Path("/mnt/e/semisynthetic_data/databases/themisto")
    clust_f = open(db_dir / "clusters.txt", "rt")
    seq_f = open(db_dir / "sequences.txt", "rt")

    df_entries = []
    for clust_line, seq_line in zip(clust_f, seq_f):
        basename = Path(seq_line.strip()).name
        if basename.endswith(".chrom.fna"):
            acc = basename[:-len(".chrom.fna")]
        elif basename.endswith(".fasta"):
            acc = basename[:-len(".fasta")]
        else:
            raise ValueError(f"Couldn't parse strain ID from line: `{seq_line}`")
        clust = clust_line.strip()
        df_entries.append({'Accession': acc, 'Cluster': clust})

    clust_f.close()
    seq_f.close()

    msweep_clust_df = pd.DataFrame(df_entries)
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


def msweep_subset_prediction(
        truth_accs,
        ground_truth,
        pred: np.ndarray,
        pred_ordering: Dict[str, int],
        clust_subset: List[str],
        msweep_clust_df: pd.DataFrame
):
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
        cluster_id = msweep_clust_df.loc[msweep_clust_df['Accession'] == sim_acc, 'Cluster'].item()
        tgt_idx = clust_indices[cluster_id]  # this will throw an error if the cluster is not phylogroup A
        SUBSET_truth[:, tgt_idx] = ground_truth[:, sim_index]
        SUBSET_true_labels[tgt_idx] = True
        
    return SUBSET_pred, SUBSET_truth, SUBSET_true_labels


def msweep_results(
    mut_ratio: str, 
    replicate: int, 
    read_depth: int, 
    trial: int, 
    abundance_bins: np.ndarray,
    lod: float = 0.002
):
    target_accs, time_points, ground_truth = load_ground_truth(mut_ratio=mut_ratio, replicate=replicate)
    pred, clusters, msweep_clust_df, runtime = extract_msweep_prediction(
        mut_ratio, replicate, read_depth, trial, time_points
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

    # lod = min(
    #     0.01044,
    #     np.min(ground_truth) * (read_depth / (read_depth + 10000000))
    # )
    # A_pred_lod, _, _ = msweep_subset_prediction(target_accs, ground_truth, pred, msweep_ordering, A_clusts, msweep_clust_df, lod=lod)

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
        'Method': 'mSWEEP',
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


def msweep_roc(mut_ratio: str, replicate: int, read_depth: int, trial: int) -> Tuple[np.ndarray, np.ndarray]:
    target_accs, time_points, ground_truth = load_ground_truth(mut_ratio=mut_ratio, replicate=replicate)
    pred, clusters, msweep_clust_df, runtime = extract_msweep_prediction(
        mut_ratio, replicate, read_depth, trial, time_points
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
    A_pred, A_truth, A_true_indicators = msweep_subset_prediction(target_accs, ground_truth, pred, msweep_ordering,
                                                                  A_clusts, msweep_clust_df)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(  # Abundance thresholding, (TxS)
        y_true=np.tile(A_true_indicators, (len(time_points), 1)).flatten(),
        y_score=A_pred.flatten(),
    )
    # fpr, tpr, thresholds = sklearn.metrics.roc_curve(  # Abundance thresholding, collapsed
    #     y_true=A_true_indicators,
    #     y_score=np.max(A_pred, axis=0),
    # )
    return fpr, tpr
