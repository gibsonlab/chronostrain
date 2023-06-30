import click
from logging import Logger
from pathlib import Path

import pandas as pd

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
    '--inference-dir', '-o', 'inference_outdir',
    type=click.Path(path_type=Path, file_okay=False, exists=True, readable=True),
    required=True,
    help="Directory containing the inference output (posterior samples & text file of strain IDs)."
)
def main(
        ctx: click.Context,
        reads_input: Path,
        inference_outdir: Path
):
    """
    Quantify evidence for each strain, per timepoint for a previous inference run.
    Outputs a pandas dataframe, using the cached marginalization log-likelihood matrices.
    """
    ctx.ensure_object(Logger)
    logger = ctx.obj

    from chronostrain.model.io import TimeSeriesReads
    from chronostrain.config import cfg
    from .helpers.evidence import quantify_evidence
    reads = TimeSeriesReads.load_from_csv(reads_input, partial_load=False)
    db = cfg.database_cfg.get_database()

    try:
        with open(inference_outdir / "strains.txt", "rt") as f:
            strains = db.get_strains([line.strip() for line in f])

        dataframes = []
        for t_idx, time_slice in enumerate(reads):
            df_t = quantify_evidence(db, reads, strains, t_idx, logger)
            dataframes.append(df_t.assign(T=time_slice.time_point))
    except FileNotFoundError as e:
        logger.error("Could not find file {}. Check whether the inference actually finished.".format(e.filename))
        exit(1)

    df_concat = pd.concat(dataframes, axis=0, ignore_index=True)
    del dataframes

    df_concat.to_feather(inference_outdir / "evidence.feather")


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    my_logger = create_logger("chronostrain.quantify")
    try:
        main(obj=my_logger)
    except Exception as e:
        my_logger.exception(e)
        exit(1)
