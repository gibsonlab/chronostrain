import click
from logging import Logger
from pathlib import Path

from ..base import option


@click.command()
@click.pass_context
@option(
    '--in-path', '-i', 'in_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="The input file path."
)
@option(
    '--out-path', '-o', 'out_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=False, readable=True),
    required=True,
    help="The output file path, to be written in fastq format.",
)
@option(
    '--read-type', '-r', 'read_type',
    type=str,
    required=True,
    help="A string token specifying what type of reads the file contains. (options: paired_1, paired_2, single)"
)
@option(
    '--quality-format', '-q', 'quality_format',
    type=str,
    required=False, default="fastq",
    help="The quality format of the input file. Token must be parsable by Bio.SeqIO."
)
@option(
    '--min-read-len', '-mr', 'min_read_len',
    type=int,
    required=False, default=35,
    help="Filters out a read if its length was less than the specified value (helps reduce spurious alignments). "
         "Ideally, a read trimming tool, such as trimmomatic, should have taken care of this step already!"
)
@option(
    '--aligner', '-al', 'aligner',
    type=str,
    required=False, default='bwa-mem2',
    help='Specify the type of aligner to use. Currently available options: bwa, bowtie2.'
)
@option(
    '--identity-threshold', '-it', 'frac_identity_threshold',
    type=float,
    required=False, default=0.975,
    help="The percent identity threshold at which to filter reads."
)
@option(
    '--error-threshold', '-et', 'error_threshold',
    type=float,
    required=False, default=1.0,
    help="The upper bound on the number of expected errors, expressed as a fraction of length of the read. "
         "A value of 1.0 disables this feature."
)
def main(
        ctx: click.Context,
        in_path: Path,
        out_path: Path,
        aligner: str,
        min_read_len: int,
        frac_identity_threshold: float,
        error_threshold: float,
        read_type: str,
        quality_format: str,
):
    """
    Filter a single read file.
    """
    ctx.ensure_object(Logger)
    logger = ctx.obj
    logger.info(f"Applying filter to `{in_path}`")

    from chronostrain.config import cfg
    from .base import Filter, create_aligner
    from chronostrain.model.io import parse_read_type

    db = cfg.database_cfg.get_database()

    # =========== Parse reads.
    filter = Filter(
        db=cfg.database_cfg.get_database(),
        min_read_len=min_read_len,
        frac_identity_threshold=frac_identity_threshold,
        error_threshold=error_threshold
    )

    read_type = parse_read_type(read_type)
    aligner_obj = create_aligner(aligner, read_type, db)
    filter.apply(
        in_path,
        out_path,
        aligner=aligner_obj,
        read_type=read_type,
        quality_format=quality_format
    )

    logger.info(f"Wrote output to {out_path}")


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    my_logger = create_logger("chronostrain.filter")
    try:
        main(obj=my_logger)
    except Exception as e:
        my_logger.exception(e)
        exit(1)
