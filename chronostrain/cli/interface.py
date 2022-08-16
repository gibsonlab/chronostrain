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
@click.pass_context
def main(ctx):
    """
    ChronoStrain (Time-Series Metagenomic Abundance Estimation)
    """
    ctx.obj = logger


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        exit(1)
