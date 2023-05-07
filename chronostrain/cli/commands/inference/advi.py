import click
from logging import Logger
from pathlib import Path

from chronostrain.util import filesystem
from ..base import option


@click.command()
@click.pass_context
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
    '--with-zeros/--without-zeros', 'with_zeros',
    is_flag=True, default=False,
    help='Specify whether to include zeros into the model..'
)
@option(
    '--iters', 'iters', type=int, default=50,
    help='The number of iterations to run per epoch.'
)
@option(
    '--epochs', 'epochs', type=int, default=1000,
    help='The number of epochs to train.'
)
@option(
    '--decay-lr', 'decay_lr', type=float, default=0.25,
    help='The multiplicative factor to apply to the learning rate based on ReduceLROnPlateau criterion.'
)
@option(
    '--lr-patience', 'lr_patience', type=int, default=5,
    help='The `patience` parameter that specifies how many epochs to tolerate no observed '
         'improvements before decaying lr.'
)
@option(
    '--min-lr', 'min_lr', type=float, default=1e-4,
    help='Stop the algorithm when the LR is below this threshold.'
)
@option(
    '--learning-rate', '-lr', 'learning_rate', type=float, default=0.001,
    help='The initial learning rate to use for optimization.'
)
@option(
    '--num-samples', '-n', 'num_samples', type=int, default=200,
    help='The number of samples to use for monte-carlo estimation of loss fn.'
)
@option(
    '--read-batch-size', '-b', 'read_batch_size', type=int, default=2500,
    help='The maximum size of each batch to use, when dividing up reads into batches.'
)
@option(
    '--correlation-mode', '-c', 'correlation_mode', type=str, default='full',
    help='The correlation mode for the posterior. Options are `full`, `strain` and `time`. '
         'For example, `strain` means that the abundance posteriors are correlated across strains, and '
         'factorized across time.'
)
@option(
    '--seed', '-s', 'seed', type=int, default=31415,
    help='Seed for randomness, specified for reproducibility.'
)
@option(
    '--true-abundances', '-truth', 'true_abundance_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    help='A CSV file path containing the ground truth relative abundances for each strain by time point. '
         'Reserved for sanity checks or debugging.'
)
@option(
    '--num-posterior-samples', '-p', type=int, default=5000,
    help='If using a variational method, specify the number of samples to generate as output.'
)
@option(
    '--allocate-fragments/--no-allocate-fragments', 'allocate_fragments',
    is_flag=True, default=True,
    help='Specify whether or not to store fragment sequences in memory '
         '(if False, will attempt to use disk-allocation instead).'
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
    '--plot-elbo/--no-plot-elbo', 'plot_elbo',
    is_flag=True, default=False,
    help='Specify whether or not to render a plot of the ELBO objective.'
)
def main(
        ctx: click.Context,
        reads_input: Path,
        out_dir: Path,
        true_abundance_path: Path,
        with_zeros: bool,
        seed: int,
        iters: int,
        epochs: int,
        decay_lr: float,
        lr_patience: int,
        min_lr: float,
        learning_rate: float,
        num_samples: int,
        read_batch_size: int,
        correlation_mode: str,
        num_posterior_samples: int,
        allocate_fragments: bool,
        plot_format: str,
        draw_training_history: bool,
        plot_elbo: bool
):
    """
    Perform posterior estimation using ADVI.
    """
    ctx.ensure_object(Logger)
    logger = ctx.obj

    logger.info("Pipeline for inference started.")
    import jax.numpy as np
    from chronostrain.config import cfg
    from chronostrain.model import Population
    from chronostrain.model.io import TimeSeriesReads
    import chronostrain.visualizations as viz
    from .initializers import load_fragments, load_fragments_dynamic, perform_advi

    # ============ Create database instance.
    db = cfg.database_cfg.get_database()

    # ============ Prepare for algorithm output.
    out_dir.mkdir(parents=True, exist_ok=True)

    elbo_path = out_dir / "elbo.{}".format(plot_format)
    animation_path = out_dir / "training.gif"
    plot_path = out_dir / "plot.{}".format(plot_format)
    samples_path = out_dir / "samples.pt"
    strains_path = out_dir / "strains.txt"
    model_out_path = out_dir / "posterior.pt"

    population = Population(strains=db.all_strains())

    # ============ Parse input reads.
    logger.info("Loading time-series read files.")
    reads = TimeSeriesReads.load_from_csv(reads_input)
    if allocate_fragments:
        fragments = load_fragments(reads, db, logger)
    else:
        fragments = load_fragments_dynamic(reads, db, logger)

    # ============ Create model instance
    solver, posterior, elbo_history, (uppers, lowers, medians) = perform_advi(
        db=db,
        population=population,
        fragments=fragments,
        reads=reads,
        with_zeros=with_zeros,
        num_epochs=epochs,
        iters=iters,
        min_lr=min_lr,
        lr_decay_factor=decay_lr,
        lr_patience=lr_patience,
        learning_rate=learning_rate,
        num_samples=num_samples,
        read_batch_size=read_batch_size,
        correlation_type=correlation_mode,
        save_elbo_history=plot_elbo,
        save_training_history=draw_training_history,
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
            model=solver.model,
            out_path=animation_path,
            upper_quantiles=uppers,
            lower_quantiles=lowers,
            medians=medians,
            elbo_history=elbo_history,
        )

    # ==== Plot the posterior.
    # Generate and save posterior samples.
    samples = posterior.sample(num_posterior_samples).detach().cpu()
    np.save(str(samples_path), samples)
    logger.info("Posterior samples saved to {}. [{}]".format(
        samples_path,
        filesystem.convert_size(samples_path.stat().st_size)
    ))
    viz.plot_vi_posterior(
        times=solver.model.times,
        population=population,
        samples=samples.cpu(),
        plot_path=plot_path,
        plot_format=plot_format,
        ground_truth_path=true_abundance_path,
        draw_legend=False
    )

    # ==== Output strain ordering.
    with open(strains_path, "w") as f:
        for strain in population.strains:
            print(strain.id, file=f)

    # ==== Save the posterior distribution.
    posterior.save(model_out_path)
    logger.debug(f"Saved model to `{model_out_path}`.")


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    my_logger = create_logger("chronostrain.inference")
    try:
        main(obj=my_logger)
    except Exception as e:
        my_logger.exception(e)
        exit(1)
