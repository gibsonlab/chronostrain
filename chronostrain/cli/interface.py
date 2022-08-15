import click
from .commands import *


@click.group(
    context_settings={'help_option_names': ['-h', '--help']},
    help="ChronoStrain CLI",
    commands={
        'filter': filter_timeseries
    }
)
def cli():
    pass
