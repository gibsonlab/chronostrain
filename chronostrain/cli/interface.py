import click
from .commands import *


@click.group(
    context_settings={'help_option_names': ['-h', '--help']},
    help="ChronoStrain (Time-Series Metagenomic Abundance Estimation)",
    commands={
        'filter': filter_timeseries,
        'filter-single': filter_single
    }
)
def cli():
    pass


if __name__ == "__main__":
    cli()
