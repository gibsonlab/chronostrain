import click
from pathlib import Path

from chronostrain.util import filesystem
from ..base import option


@click.command()
@option(
    '--reads', '-r', 'reads_input',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the reads input CSV file."
)
@option(
    '--out-dir', '-o', 'out_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="The directory to save all outputs to."
)
@option(
    '--strain-subset', '-s', 'strain_subset_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=False, readable=True),
    required=False, default=None,
    help="A text file specifying a subset of database strain IDs to perform filtering with; "
         "a TSV file containing one ID per line, optionally with a second column for metadata.",
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
def main(
        reads_input: Path,
        out_dir: Path,
        strain_subset_path: Path,
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
        plot_elbo: bool
):
    """
    Perform posterior estimation using ADVI.
    """
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli.advi")

    logger.info("Pipeline for algs started.")
    import jax.numpy as np
    from chronostrain.config import cfg
    from chronostrain.model import StrainCollection
    from chronostrain.model.io import TimeSeriesReads
    import chronostrain.visualizations as viz
    from .helpers import perform_advi

    # ============ Create database instance.
    db = cfg.database_cfg.get_database()
    if strain_subset_path is not None:
        with open(strain_subset_path, "rt") as f:
            strain_collection = StrainCollection(
                [db.get_strain(line.strip().split('\t')[0]) for line in f if not line.startswith("#")],
                db.signature
            )
    else:
        strain_collection = StrainCollection(db.all_strains(), db.signature)

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
        db=db,
        population=strain_collection,
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
        population=solver.gaussian_prior.strain_collection,
        samples=samples,
        plot_path=plot_path,
        plot_format=plot_format,
        ground_truth_path=true_abundance_path,
        draw_legend=False
    )

    # ==== Output strain ordering.
    with open(strains_path, "w") as f:
        for strain in solver.gaussian_prior.strain_collection.strains:
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


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    main_logger = create_logger("chronostrain.MAIN")
    try:
        main()
    except Exception as e:
        main_logger.exception(e)
        exit(1)
