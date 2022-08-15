import click
import commands


@click.group(
    context_settings={'help_option_names': ['-h', '--help']},
    help="ChronoStrain (Time-Series Metagenomic Abundance Estimation)",
    commands={
        'filter': commands.filter_timeseries,
        'filter-single': commands.filter_single
    }
)
def cli():
    pass


if __name__ == "__main__":
    cli()
