import click
from pathlib import Path

from ..base import option


@click.command()
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
    '--strain-subset', '-s', 'strain_subset_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=False, readable=True),
    required=False, default=None,
    help="A text file specifying a subset of database strain IDs to perform filtering with; "
         "a TSV file containing one ID per line, optionally with a second column for metadata.",
)
@option(
    '--read-type', '-r', 'read_type_str',
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
        in_path: Path,
        out_path: Path,
        strain_subset_path: Path,
        aligner: str,
        min_read_len: int,
        frac_identity_threshold: float,
        error_threshold: float,
        read_type_str: str,
        quality_format: str,
):
    """
    (Non-essential) Filter a single read file, instead of an entire timeseries.
    """
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli.filter_single")
    logger.info(f"Applying filter to `{in_path}`")

    from chronostrain.config import cfg
    from .base import Filter, create_aligner
    from chronostrain.model import ReadType
    from chronostrain.model import StrainCollection

    db = cfg.database_cfg.get_database()
    if strain_subset_path is not None:
        with open(strain_subset_path, "rt") as f:
            strain_collection = StrainCollection(
                [db.get_strain(line.strip().split('\t')[0]) for line in f if not line.startswith("#")],
                db.signature
            )
        logger.info("Loaded list of {} strains.".format(len(strain_collection)))
    else:
        strain_collection = StrainCollection(db.all_strains(), db.signature)
        logger.info("Using complete collection of {} strains from database.".format(len(strain_collection)))


    # =========== Parse reads.
    filter = Filter(
        db=cfg.database_cfg.get_database(),
        strain_collection=strain_collection,
        min_read_len=min_read_len,
        frac_identity_threshold=frac_identity_threshold,
        error_threshold=error_threshold
    )

    read_type = ReadType.parse_from_str(read_type_str)
    aligner_obj = create_aligner(aligner, read_type, strain_collection.multifasta_file)
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
    main_logger = create_logger("chronostrain.MAIN")
    try:
        main()
    except Exception as e:
        main_logger.exception(e)
        exit(1)
