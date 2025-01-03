from .chronostrain import *


def extract_chronostrain_indiv_prediction(
        mut_ratio: str, replicate: int, read_depth: int, trial: int, time_index: int, subdir_name: str = 'chronostrain'
) -> Tuple[GaussianWithGlobalZerosPosteriorDense, List[str], Dict, int]:
    output_basedir = trial_dir(mut_ratio, replicate, read_depth, trial) / 'output'
    output_dir = output_basedir / subdir_name / f'timepoint_{time_index}'
    if not output_dir.exists():
        raise FileNotFoundError(
            f"Output for replicate={replicate}, read_depth={read_depth}, trial={trial}, time_index={time_index} does not exist.")

    # samples = np.load(str(output_dir / 'samples.npy'))
    strains = parse_chronostrain_strains(output_dir / 'strains.txt')

    n_time_points = 1  # this is equal to 1 for chronostrain_indiv, for code readability
    posterior = GaussianWithGlobalZerosPosteriorDense(len(strains), n_time_points, cfg.engine_cfg.dtype)
    posterior.load(Path(output_dir / "posterior.{}.npz".format(cfg.engine_cfg.dtype)))

    adhoc_clusters = parse_adhoc_clusters(output_dir / 'adhoc_cluster.txt')
    runtime = parse_runtime(output_dir / 'inference_runtime.txt')
    return posterior, strains, adhoc_clusters, runtime


def chronostrain_indiv_output_slice(
    mut_ratio: str,
    replicate: int, 
    read_depth: int, 
    trial: int,
    time_index: int,
    subdir_name: str = 'chronostrain_indiv', 
    posterior_threshold: float = 0.9901,
    prior_p: float = 0.001
):
    truth_accs, time_points, ground_truth = load_ground_truth(mut_ratio=mut_ratio, replicate=replicate)
    
    """ First, specific to chronostrain_indiv: Take the slice corresponding to this timepoint. """
    time_points = [time_points[time_index]]
    ground_truth = ground_truth[[time_index], :]
    
    posterior, strains, adhoc_clusters, runtime = extract_chronostrain_indiv_prediction(mut_ratio, replicate, read_depth, trial, time_index, subdir_name=subdir_name)

    strain_to_idx = {s: i for i, s in enumerate(strains)}
    chronostrain_clustering_df = load_chronostrain_cluster(Path('/mnt/e/ecoli_db/chronostrain_files/ecoli.clusters.txt'))
    chronostrain_clustering_df = chronostrain_clustering_df.assign(
        Adhoc=chronostrain_clustering_df['Cluster'].map(adhoc_clusters)
    )

    """ Extract phylogroup A prediction/truth. """
    # First, check that the four target genomes are in distinct clusters.
    sim_clusts = chronostrain_clustering_df.loc[
        chronostrain_clustering_df['Accession'].isin(set(truth_accs)), 'Cluster'
    ]
    if len(truth_accs) != len(pd.unique(sim_clusts)):
        raise ValueError(
            "Resulting clustering does not separate the {} target accessions for replicate {}, read_depth {}, trial {}".format(
                len(truth_accs), replicate, read_depth, trial))

    """ Extract samples. """
    n_samples = 5000
    sample_dir = Path() / 'samples' / f'{subdir_name}' / f'mutratio_{mut_ratio}' / f'replicate_{replicate}' / f'reads_{read_depth}' / f'trial_{trial}' / f'timepoint_{time_index}'  # relative dir
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
        g_samples, z_samples, prior_p,
        phylogroup_A_clusts,
        strain_to_idx,
        chronostrain_clustering_df,
        posterior_threshold=posterior_threshold
    )
    if np.sum(A_pred_indicators > posterior_threshold) == 0:
        print(f"mut_ratio = {mut_ratio} | replicate = {replicate} | read_depth = {read_depth} | trial = {trial} | Timepoint {time_index}: [A] Found empty prediction through posterior filtering; defaulting to uniform profile.")
        A_pred = np.ones(A_pred.shape) / A_pred.shape[-1]

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
    if np.sum(sim_pred_indicators > posterior_threshold) == 0:
        print(f"mut_ratio = {mut_ratio} | replicate = {replicate} | read_depth = {read_depth} | trial = {trial} | Timepoint {time_index}: [sim-only] Found empty prediction through posterior filtering; defaulting to uniform profile.")
        sim_pred = np.ones(sim_pred.shape) / sim_pred.shape[-1]

    # ================ Metric evaluation
    # Median trajectories.
    A_median_traj = np.median(A_pred, axis=1)
    A_median_traj = A_median_traj / np.sum(A_median_traj, axis=-1, keepdims=True)  # renormalize
    sim_median_traj = np.median(sim_pred, axis=1)
    sim_median_traj = sim_median_traj / np.sum(sim_median_traj, axis=-1, keepdims=True)
    return A_pred_indicators, A_median_traj, sim_median_traj, A_true_indicators, A_truth, sim_truth, runtime


def chronostrain_indiv_results(
    mut_ratio: str,
    replicate: int, 
    read_depth: int, 
    trial: int,
    abundance_bins: np.ndarray,
    subdir_name: str = 'chronostrain_indiv', 
    posterior_threshold: float = 0.9901,
    prior_p: float = 0.001
):
    truth_accs, time_points, ground_truth = load_ground_truth(mut_ratio=mut_ratio, replicate=replicate)

    chronostrain_clustering_df = load_chronostrain_cluster(Path('/mnt/e/ecoli_db/chronostrain_files/ecoli.clusters.txt'))
    n_times = len(time_points)
    n_A_clusts = len(pd.unique(
        chronostrain_clustering_df.loc[chronostrain_clustering_df['ClusterPhylogroup'] == 'A', 'Cluster']
    ))
    
    n_sim_clusts = ground_truth.shape[-1]

    A_pred_indicators = np.empty((n_times, n_A_clusts), dtype=float)
    A_true_indicators = np.empty((n_times, n_A_clusts), dtype=float)
    A_truth = np.empty((n_times, n_A_clusts), dtype=ground_truth.dtype)
    sim_truth = np.empty((n_times, n_sim_clusts), dtype=ground_truth.dtype)
    A_median_traj = np.empty((n_times, n_A_clusts), dtype=ground_truth.dtype)
    sim_median_traj = np.empty((n_times, n_sim_clusts), dtype=ground_truth.dtype)
    runtimes = []

    chronostrain_original_output_dir = trial_dir(mut_ratio, replicate, read_depth, trial) / 'output' / 'chronostrain'
    try:
        filter_runtime = parse_runtime(chronostrain_original_output_dir / 'filter_runtime.txt')
        runtimes.append(filter_runtime)
    except FileNotFoundError as e:
        print("Couldn't find filter runtime file while parsing chronostrain_indiv (loc={}). Defaulting to NaN.".format(
            chronostrain_original_output_dir
        ))
    
    for t_idx in range(len(time_points)):
        slice_A_pred_indicators, slice_A_median_traj, slice_sim_median_traj, slice_A_true_indicators, slice_A_truth, slice_sim_truth, slice_runtime = chronostrain_indiv_output_slice(
            mut_ratio, replicate, read_depth, trial,
            t_idx,
            subdir_name, posterior_threshold, prior_p
        )
        if n_A_clusts != len(slice_A_pred_indicators):
            raise ValueError("Got {} phylogroup A clusters from slice output, but expected {}".format(len(slice_A_pred_indicators), n_A_clusts))

        runtimes.append(slice_runtime)
        A_median_traj[t_idx] = slice_A_median_traj[0]
        sim_median_traj[t_idx] = slice_sim_median_traj[0]
        A_truth[t_idx] = slice_A_truth[0]
        sim_truth[t_idx] = slice_sim_truth[0]
        A_pred_indicators[t_idx] = slice_A_pred_indicators
        A_true_indicators[t_idx] = slice_A_true_indicators

    # ================ Metric evaluation
    auroc_abund = sklearn.metrics.roc_auc_score(  # Abundance thresholding
        y_true=A_true_indicators.flatten(),
        y_score=A_median_traj.flatten(),
    )
    auroc_posterior = sklearn.metrics.roc_auc_score(  # posterior score
        y_true=A_true_indicators.flatten(),
        y_score=A_pred_indicators.flatten()
    )

    tv_err_sim = tv_error(sim_median_traj, sim_truth)
    tv_err_A = tv_error(A_median_traj, A_truth)

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
        'Method': 'ChronoStrain_Indiv',
        'NumClusters': n_A_clusts,
        'TVErrorSim': tv_err_sim,
        'TVErrorA': tv_err_A,
        'RMSErrorSim': rms_error_sim,
        'RMSErrorA': rms_error_A,
        'AUROC_abundance': auroc_abund,
        'AUROC_posterior': auroc_posterior,
        'RankCorrelationSim': rank_corr_sim,
        'RankCorrelationA': rank_corr_A,
        'Runtime': np.sum(runtimes)
    } | {
        f'RMSErrorSim_Bin{i}': binned_rms
        for i, binned_rms in enumerate(binned_rms_error_sim)
    } | {
        f'RMSErrorSim_Strain{i}': _rms
        for i, _rms in enumerate(split_rms_sim)
    }