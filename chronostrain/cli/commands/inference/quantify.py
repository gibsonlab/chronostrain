import click
from logging import Logger
from pathlib import Path

from chronostrain.cli.commands.base import option


@click.command()
@click.pass_context
@option(
    '--reads', '-r', 'reads_input',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the reads input CSV file."
)
@option(
    '--inference-dir', '-o', 'inference_outdir',
    type=click.Path(path_type=Path, file_okay=False, exists=True, readable=True),
    required=True,
    help="Directory containing the inference output (posterior samples & text file of strain IDs)."
)
@option(
    '--batch-size', '-b', 'read_batch_size',
    type=int, required=False, default=5000,
    help="The batch size to use; a lower value uses less memory but computation takes longer."
)
@option(
    '--allocate-fragments/--no-allocate-fragments', 'allocate_fragments',
    is_flag=True, default=True,
    help='Required for initialization. '
         'Specify whether or not to store fragment sequences in memory. '
         'To avoid re-calculation, use the same argument as what was used for inference.'
)
def main(
        ctx: click.Context,
        reads_input: Path,
        inference_outdir: Path,
        read_batch_size: int,
        allocate_fragments: bool
):
    """
    Quantify evidence for each strain, per timepoint for a previous inference run.
    Outputs a pandas dataframe, using the cached marginalization log-likelihood matrices.
    """
    ctx.ensure_object(Logger)
    logger = ctx.obj

    from chronostrain.model.io import TimeSeriesReads
    from chronostrain.model import Population
    from chronostrain.config import cfg
    from .helpers import create_model, quantify_evidence, load_fragments, load_fragments_dynamic

    reads = TimeSeriesReads.load_from_csv(reads_input)
    db = cfg.database_cfg.get_database()

    if allocate_fragments:
        fragments = load_fragments(reads, db, logger)
    else:
        fragments = load_fragments_dynamic(reads, db, logger)
    model = create_model(
        population=Population(strains=db.all_strains()),
        source_reads=reads,
        fragments=fragments,
        time_points=[time_slice.time_point for time_slice in reads],
        disable_quality=not cfg.model_cfg.use_quality_scores,
        db=db,
        logger=logger
    )

    with open(inference_outdir / "strains.txt", "rt") as f:
        target_strains = db.get_strains([line.strip() for line in f])

    df = quantify_evidence(
        db, model, reads, logger, target_strains, read_batch_size=read_batch_size
    )
    df.to_feather(inference_outdir / "evidence.feather")


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    my_logger = create_logger("chronostrain.quantify")
    try:
        main(obj=my_logger)
    except Exception as e:
        my_logger.exception(e)
        exit(1)
