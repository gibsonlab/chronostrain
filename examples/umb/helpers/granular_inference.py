from typing import List, Dict, Tuple

import click
from pathlib import Path
from click import option

import numpy as np
import scipy.special
from chronostrain.database import StrainDatabase
from chronostrain.model import Strain, TimeSeriesReads
from chronostrain.util import filesystem
from chronostrain.inference import GaussianWithGumbelsPosterior
from chronostrain.config import cfg


@click.command()
@option(
    '--reads', '-r', 'reads_input',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the reads input CSV file."
)
@option(
    '--coarse-dir', '-c', 'coarse_inference_outdir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="The directory that contains the result of the ``coarse'' inference."
)
@option(
    '--granular-json', '-gj', 'granular_db_json',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="The JSON file that points to the desired granular database specification."
)
@option(
    '--out-dir', '-o', 'out_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="The directory to save all outputs to."
)
@option(
    '--with-zeros/--without-zeros', 'with_zeros',
    is_flag=True, default=False,
    help='Specify whether to include zeros into the model.'
)
@option(
    '--prior-p', 'prior_p', type=float, default=0.5,
    help='The prior bias for the indicator variables, where bias = P(strain included in model).'
)
@option(
    '--prune-strains/--dont-prune-strains', 'prune_strains',
    is_flag=True, default=False,
    help='Specify whether to prune the input database strains based on the read_frags.'
)
@option(
    '--iters', 'iters', type=int, default=100,
    help='The number of iterations to run per epoch.'
)
@option(
    '--epochs', 'epochs', type=int, default=10000,
    help='The number of epochs to train.'
)
@option(
    '--decay-lr', 'decay_lr', type=float, default=0.1,
    help='The multiplicative factor to apply to the learning rate based on ReduceLROnPlateau criterion.'
)
@option(
    '--lr-patience', 'lr_patience', type=int, default=5,
    help='The `patience` parameter that specifies how many epochs to tolerate no observed '
         'improvements before decaying lr.'
)
@option(
    '--min-lr', 'min_lr', type=float, default=1e-7,
    help='Stop the algorithm when the LR is below this threshold.'
)
@option(
    '--loss-tol', 'loss_tol', type=float, default=1e-7,
    help='Stop the algorithm when the relative change in ELBO is smaller than this fraction.'
)
@option(
    '--learning-rate', '-lr', 'learning_rate', type=float, default=0.0005,
    help='The initial learning rate to use for optimization.'
)
@option(
    '--num-samples', '-n', 'num_samples', type=int, default=100,
    help='The number of samples to use for monte-carlo estimation of loss fn.'
)
@option(
    '--read-batch-size', '-b', 'read_batch_size', type=int, default=10000,
    help='The maximum size of each batch to use, when dividing up reads into batches.'
)
@option(
    '--adhoc-corr-threshold', '-ct', 'adhoc_corr_threshold', type=float, default=0.99,
    help='Just before running inference, will attempt to estimate clusters of strains that are indistinguishable '
         'from the provided reads, using an empirical correlation matrix of the marginalized read-strain likelihoods. '
         'Strains will be ad-hoc clustered together if they have correlation at least 0.99. '
         'After inference, this clustering is output to the file `adhoc_cluster.txt`.'
)
@option(
    '--correlation-mode', '-c', 'correlation_mode', type=str, default='full',
    help='The correlation mode for the posterior. Options are `full`, `strain` and `time`. '
         'For example, `strain` means that the abundance posteriors are correlated across strains, and '
         'factorized across time.'
)
@option(
    '--true-abundances', '-truth', 'true_abundance_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    help='A CSV file path containing the ground truth relative abundances for each strain by time point. '
         'Reserved for sanity checks or debugging.'
)
@option(
    '--num-posterior-samples', '-p', 'num_output_samples', type=int, default=5000,
    help='If using a variational method, specify the number of samples to generate as output.'
)
@option(
    '--plot-format', 'plot_format', type=str, default='pdf',
    help='The format to use for saving posterior plots.'
)
@option(
    '--draw-training/--no-draw-training', 'draw_training_history',
    is_flag=True, default=False,
    help='Specify whether or not to render a GIF of the posterior mid-training.'
)
@option(
    '--accumulate-gradients/--dont-accumulate-gradients', 'accumulate_gradients',
    is_flag=True, default=False,
    help='Specify whether to accumulate gradients (for saving memory, at slight cost of runtime).'
         'Results are expected to be completely identical in either mode; this just changes the '
         'way the ELBO function is JIT-compiled.'
)
@option(
    '--plot-elbo/--no-plot-elbo', 'plot_elbo',
    is_flag=True, default=False,
    help='Specify whether or not to render a plot of the ELBO objective.'
)
@option(
    '--abund-lb', '-lb', 'abund_lb', type=float, default=0.05,
    help='The (database-normalized) abund lower bound to determine presence/absence from sample. '
         'Only used for parsing the coarse output.'
)
@option(
    '--bayes-factor', '-bf', 'bf_threshold', type=float, default=100000.0,
    help='The Bayes factor threshold for parsing the coarse output.'
)
def main(
        reads_input: Path,
        coarse_inference_outdir: Path,
        granular_db_json: Path,
        out_dir: Path,
        true_abundance_path: Path,
        with_zeros: bool,
        prior_p: float,
        prune_strains: bool,
        iters: int,
        epochs: int,
        decay_lr: float,
        lr_patience: int,
        min_lr: float,
        loss_tol: float,
        learning_rate: float,
        num_samples: int,
        read_batch_size: int,
        adhoc_corr_threshold: float,
        correlation_mode: str,
        num_output_samples: int,
        plot_format: str,
        draw_training_history: bool,
        accumulate_gradients: bool,
        plot_elbo: bool,
        abund_lb: float,
        bf_threshold: float
):
    """
    Perform posterior estimation using ADVI.
    """
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli.advi")

    logger.info("Pipeline for algs started.")
    import jax.numpy as np
    from chronostrain.config import cfg
    from chronostrain.database import JSONParser
    from chronostrain.model import Population
    from chronostrain.model.io import TimeSeriesReads
    from chronostrain.inference import GaussianStrainCorrelatedWithGlobalZerosPosterior
    from chronostrain.database import QueryNotFoundError
    import chronostrain.visualizations as viz
    from chronostrain.cli.commands.inference.helpers import perform_advi

    # ============ Extract the coarse inference results.
    coarse_db = cfg.database_cfg.get_database()
    coarse_inference_strain_ids = extract_coarse_inference(
        filt_reads_path=reads_input,
        coarse_inference_outdir=coarse_inference_outdir,
        coarse_db=coarse_db,
        posterior_class=GaussianStrainCorrelatedWithGlobalZerosPosterior,
        bf_threshold=bf_threshold,
        prior_p=prior_p,
        abund_lb=abund_lb
    )

    # ============ Create database instance.
    granular_db = JSONParser(
        data_dir=cfg.database_cfg.data_dir,
        entries_file=granular_db_json,
        marker_max_len=cfg.database_cfg.parser_kwargs['marker_max_len']
    ).parse()

    coarse_inference_strains = []
    for s_id in coarse_inference_strain_ids:
        try:
            s = granular_db.get_strain(s_id)
            coarse_inference_strains.append(s)  # Some of these might still be clustered together.
        except QueryNotFoundError:
            continue

    # ============ Prepare for algorithm output.
    out_dir.mkdir(parents=True, exist_ok=True)

    elbo_path = out_dir / "elbo.{}".format(plot_format)
    animation_path = out_dir / "training.gif"
    plot_path = out_dir / "plot.{}".format(plot_format)
    samples_path = out_dir / "samples.npy"
    strains_path = out_dir / "strains.txt"
    model_out_path = out_dir / "posterior.{}.npz".format(cfg.engine_cfg.dtype)

    # ============ Parse input reads.
    logger.info("Loading time-series read files from {}".format(reads_input))
    reads = TimeSeriesReads.load_from_file(reads_input)

    if reads.total_number_reads() == 0:
        logger.info("No filtered reads found. Exiting.")
        return 0

    # ============ Create model instance
    solver, posterior, elbo_history, (uppers, lowers, medians) = perform_advi(
        db=granular_db,
        population=Population(strains=coarse_inference_strains),
        reads=reads,
        with_zeros=with_zeros,
        prior_p=prior_p,
        initialize_with_map=False,
        prune_strains=prune_strains,
        num_epochs=epochs,
        iters=iters,
        min_lr=min_lr,
        loss_tol=loss_tol,
        lr_decay_factor=decay_lr,
        lr_patience=lr_patience,
        learning_rate=learning_rate,
        num_samples=num_samples,
        read_batch_size=read_batch_size,
        adhoc_corr_threshold=adhoc_corr_threshold,
        correlation_type=correlation_mode,
        save_elbo_history=plot_elbo,
        save_training_history=draw_training_history,
        accumulate_gradients=accumulate_gradients,
        logger=logger
    )

    if plot_elbo:
        viz.plot_elbo_history(
            elbos=elbo_history,
            out_path=elbo_path,
            plot_format=plot_format
        )

    if draw_training_history:
        viz.plot_training_animation(
            model=solver.gaussian_prior,
            out_path=animation_path,
            upper_quantiles=uppers,
            lower_quantiles=lowers,
            medians=medians,
            elbo_history=elbo_history,
        )

    # ==== Plot the posterior.
    # Generate and save posterior samples.
    samples = posterior.abundance_sample(num_output_samples)
    np.save(str(samples_path), samples.astype(np.float32))
    logger.info("Posterior samples saved to {}. [{}]".format(
        samples_path,
        filesystem.convert_size(samples_path.stat().st_size)
    ))
    viz.plot_vi_posterior(
        times=solver.gaussian_prior.times,
        population=solver.gaussian_prior.population,
        samples=samples,
        plot_path=plot_path,
        plot_format=plot_format,
        ground_truth_path=true_abundance_path,
        draw_legend=False
    )

    # ==== Output strain ordering.
    with open(strains_path, "w") as f:
        for strain in solver.gaussian_prior.population.strains:
            print(strain.id, file=f)

    # ==== Report any ad-hoc clustering.
    if len(solver.adhoc_clusters) > 0:
        with open(out_dir / "adhoc_cluster.txt", "wt") as f:
            for rep, clust in solver.adhoc_clusters.items():
                print("{}:{}".format(
                    rep,
                    ",".join(clust)
                ), file=f)

    # ==== Save the posterior distribution.
    posterior.save(model_out_path)
    logger.info(f"Saved model to `{model_out_path}`.")


def parse_adhoc_clusters(db: StrainDatabase, txt_file: Path) -> Dict[str, Strain]:
    clust = {}
    with open(txt_file, "rt") as f:
        for line in f:
            tokens = line.strip().split(":")
            rep = tokens[0]
            members = tokens[1].split(",")
            for member in members:
                clust[member] = db.get_strain(rep)
    return clust


def parse_strains(db: StrainDatabase, strain_txt: Path):
    with open(strain_txt, 'rt') as f:
        return [
            db.get_strain(l.strip())
            for l in f
        ]

def extract_coarse_inference(
        filt_reads_path: Path,
        coarse_inference_outdir: Path,
        coarse_db: StrainDatabase,
        posterior_class,
        bf_threshold: float,
        prior_p: float,
        abund_lb: float
) -> List[str]:
    adhoc_clusters: Dict[str, Strain] = parse_adhoc_clusters(coarse_db, coarse_inference_outdir / "adhoc_cluster.txt")
    inference_strains: List[Strain] = parse_strains(coarse_db, coarse_inference_outdir / 'strains.txt')
    display_strains: List[Strain] = list(coarse_db.get_strain(x) for x in adhoc_clusters.keys())

    reads = TimeSeriesReads.load_from_file(filt_reads_path)
    time_points = np.array([reads_t.time_point for reads_t in reads], dtype=float)
    posterior = posterior_class(
        len(inference_strains),
        len(time_points),
        cfg.engine_cfg.dtype
    )
    posterior.load(Path(coarse_inference_outdir / "posterior.{}.npz".format(cfg.engine_cfg.dtype)))

    posterior_p, posterior_samples = posterior_with_bf_threshold(
        posterior=posterior,
        inference_strains=inference_strains,
        output_strains=display_strains,
        adhoc_clustering=adhoc_clusters,
        bf_threshold=bf_threshold,
        prior_p=prior_p
    )

    strains_to_output = {}
    for t_idx, t in enumerate(time_points):
        for strain_idx, strain in enumerate(display_strains):
            filt_relabunds = posterior_samples[t_idx, :, strain_idx]
            if np.median(filt_relabunds) > abund_lb:
                strains_to_output[strain.id] = strain

    strain_full_ids = set()
    for s_id, strain in strains_to_output.items():
        for member_str in strain.metadata.cluster:
            member_id = member_str.split("(")[0]
            strain_full_ids.add(member_id)
    return sorted(strain_full_ids)


def posterior_with_bf_threshold(
        posterior: GaussianWithGumbelsPosterior,
        inference_strains: List[Strain],
        output_strains: List[Strain],
        adhoc_clustering: Dict[str, Strain],
        bf_threshold: float,
        prior_p: float
) -> Tuple[Dict[str, float], np.ndarray]:
    # Raw random samples.
    n_samples = 5000
    rand = posterior.random_sample(n_samples)
    g_samples = np.array(
        posterior.reparametrized_gaussians(rand['std_gaussians'], posterior.get_parameters()))  # T x N x S
    z_samples = np.array(posterior.reparametrized_zeros(rand['std_gumbels'], posterior.get_parameters()))
    # print(posterior.get_parameters())# N x S

    n_times = g_samples.shape[0]
    n_inference_strains = g_samples.shape[-1]
    assert n_inference_strains == len(inference_strains)

    # Calculate bayes factors.
    posterior_inclusion_p = scipy.special.expit(-posterior.get_parameters()['gumbel_diff'])
    # print(posterior_inclusion_p)
    posterior_inclusion_bf = (posterior_inclusion_p / (1 - posterior_inclusion_p)) * ((1 - prior_p) / prior_p)

    # Calculate abundance estimates using BF thresholds.
    indicators = np.full(n_inference_strains, fill_value=False, dtype=bool)
    indicators[posterior_inclusion_bf > bf_threshold] = True
    print("{} of {} inference strains passed BF Threshold > {}".format(np.sum(indicators), n_inference_strains,
                                                                       bf_threshold))

    log_indicators = np.empty(n_inference_strains, dtype=float)
    log_indicators[indicators] = 0.0
    log_indicators[~indicators] = -np.inf
    pred_abundances_raw = scipy.special.softmax(g_samples + np.expand_dims(log_indicators, axis=[0, 1]), axis=-1)

    # Unwind the adhoc grouping.
    pred_abundances = np.zeros(shape=(n_times, n_samples, len(output_strains)), dtype=float)
    adhoc_indices = {s.id: i for i, s in enumerate(inference_strains)}
    output_indices = {s.id for s in output_strains}
    for s_idx, s in enumerate(output_strains):
        adhoc_rep = adhoc_clustering[s.id]
        adhoc_idx = adhoc_indices[adhoc_rep.id]
        adhoc_clust_ids = set(s_ for s_, clust in adhoc_clustering.items() if clust.id == adhoc_rep.id)
        adhoc_sz = len(adhoc_clust_ids.intersection(output_indices))
        # if adhoc_sz > 1:
        #     print(f"{s.id} [{s.metadata.genus} {s.metadata.species}, {s.name}] --> adhoc sz = {adhoc_sz} (Adhoc Cluster {adhoc_rep.id} [{adhoc_rep.metadata.genus} {adhoc_rep.metadata.species}, {adhoc_rep.name}])")
        pred_abundances[:, :, s_idx] = pred_abundances_raw[:, :, adhoc_idx] / adhoc_sz
    return {
        s.id: posterior_inclusion_p[
            adhoc_indices[adhoc_clustering[s.id].id]
        ]
        for i, s in enumerate(output_strains)
    }, pred_abundances


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    main_logger = create_logger("chronostrain.MAIN")
    try:
        main()
    except Exception as e:
        main_logger.exception(e)
        exit(1)
