import click
from logging import Logger
from .commands import *


@click.group(
    context_settings={'help_option_names': ['-h', '--help']},
    commands={
        'filter': filter_timeseries,
        'filter-single': filter_single,
        'advi': run_advi
    }
)
@click.pass_context
def main(ctx):
    """
    ChronoStrain (Time-Series Metagenomic Abundance Estimation)
    """
    ctx.ensure_object(Logger)


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli")
    try:
        main(obj=logger)
    except BaseException as e:
        logger.exception(e)
        exit(1)
