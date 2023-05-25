from typing import Optional
from pathlib import Path
import os

import click
from .commands import *
from chronostrain.logging import create_logger
logger = create_logger("chronostrain.cli")


@click.group(
    context_settings={
        'help_option_names': ['-h', '--help'],
        'max_content_width': 120
    },
    commands={
        'filter': filter_timeseries,
        'filter-single': filter_single,
        'advi': run_advi,
        'make-db': make_db,
        'prune-pickle': prune_pickle
    }
)
@click.option(
    '--config', '-c', 'config_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=False,
    help="The path to a chronostrain INI configuration file."
)
@click.option(
    '--profile-jax/--dont-profile-jax', 'jax_profile',
    is_flag=True, default=False,
    help='Specify whether to profile JAX memory usage using jax-smi.'
)
@click.pass_context
def main(ctx, config_path: Optional[Path], jax_profile: bool = False):
    """
    ChronoStrain (Time-Series Abundance Estimation from Metagenomic Shotgun Sequencing)
    Contact: Younhun Kim (younhun@mit.edu)
    """
    ctx.obj = logger
    if config_path is not None:
        os.environ['CHRONOSTRAIN_INI'] = str(config_path)
    if jax_profile:
        from jax_smi import initialise_tracking
        initialise_tracking()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        exit(1)
