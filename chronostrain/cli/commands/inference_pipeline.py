import click
from pathlib import Path

from .base import option
from .filter import filter_timeseries
from .inference import run_advi


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
    help="The directory to which the filtered reads/CSV table will be saved.",
)
@option(
    '--read-batch-size', '-b', 'read_batch_size', type=int, default=10000,
    help='The maximum size of each batch to use, when dividing up reads into batches.'
)
@option(
    '--correlation-mode', '-c', 'correlation_mode', type=str, default='full',
    help='The correlation mode for the posterior. Options are `full`, `strain` and `time`. '
         'For example, `strain` means that the abundance posteriors are correlated across strains, and '
         'factorized across time.'
)
@option(
    '--accumulate-gradients/--dont-accumulate-gradients', 'accumulate_gradients',
    is_flag=True, default=False,
    help='Specify whether to accumulate gradients (for saving memory, at slight cost of runtime).'
         'Results are expected to be completely identical in either mode; this just changes the '
         'way the ELBO function is JIT-compiled.'
)
def run_inference_pipeline(
        ctx: click.Context,
        reads_input: Path,
        out_dir: Path,
        read_batch_size: int,
        correlation_mode: str,
        accumulate_gradients: bool
):
    """
    (Non-essential)
    Run the pipeline (filter + advi) with mainly default settings.
    To use non-standard settings for any of the steps, one must run the subcommands separately/manually.
    """

    """
    Note: This implementation uses ctx.invoke. This is an anti-pattern in Click, but it is the easiest way
    to glue two subcommands together with specialized modification of arguments.

    (Is there a better way to do this through pure python? (e.g. w/o bash scripts; as platform-independent as possible))
    """
    filt_reads_dir = out_dir / 'filtered'
    filt_output = filt_reads_dir / 'filtered_reads.csv'
    # Equivalent to filter_timeseries(...) if it didn't have the click.command decorator
    ctx.invoke(
        filter_timeseries,
        reads_input=reads_input,
        out_dir=filt_reads_dir,
        output_filename=filt_output.name
    )

    inference_dir = out_dir / 'inference'
    # Equivalent to run_advi(...) if it didn't have the click.command decorator
    ctx.invoke(
        run_advi,
        reads_input=filt_output,
        out_dir=inference_dir,
        read_batch_size=read_batch_size,
        correlation_mode=correlation_mode,
        accumulate_gradients=accumulate_gradients
    )


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    main_logger = create_logger("chronostrain.MAIN")
    try:
        run_inference_pipeline()
    except Exception as e:
        main_logger.exception(e)
        exit(1)
