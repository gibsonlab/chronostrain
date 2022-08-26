from typing import Optional
from pathlib import Path
import os

import click
from .commands import *
from chronostrain.logging import create_logger
logger = create_logger("chronostrain.cli")


@click.group(
    context_settings={'help_option_names': ['-h', '--help']},
    commands={
        'filter': filter_timeseries,
        'filter-single': filter_single,
        'advi': run_advi
    }
)
@click.option(
    '--config', '-c', 'config_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=False,
    help="The path to a chronostrain INI configuration file."
)
@click.pass_context
def main(ctx, config_path: Optional[Path]):
    """
    ChronoStrain (Time-Series Metagenomic Abundance Estimation)
    Contact: Younhun Kim (younhun@mit.edu)
    """
    ctx.obj = logger
    if config_path is not None:
        os.environ['CHRONOSTRAIN_INI'] = str(config_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        exit(1)
