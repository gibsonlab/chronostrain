import click
import re
import csv
from logging import Logger
from pathlib import Path
from typing import List, Tuple, Optional

from ..base import option


@click.command()
@option(
    '--reads', '-r', 'reads_input',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the reads input CSV file."
)
@option(
    '--out-dir', '-o', 'out_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="The directory to which the filtered reads/CSV table will be saved.",
)
@option(
    '--strain-subset', '-s', 'strain_subset_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=False, readable=True),
    required=False, default=None,
    help="A text file specifying a subset of database strain IDs to perform filtering with; "
         "a TSV file containing one ID per line, optionally with a second column for metadata.",
)
@option(
    '--output-filename', '-f', 'output_filename',
    type=str,
    required=False, default=None,
    help="The filename of the target CSV file indexing all filtered reads. If not specified, "
         "will default to `filtered_<input_file>`. Parent directory is always out_dir."
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
    required=False, default='bowtie2',
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
@option(
    '--attach-sample-ids', 'attach_sample_ids',
    is_flag=True, default=False,
    help='Specify whether to attach sample IDs to the fasta filename. Useful if, for some reason, '
         'the input fasta files have non-unique names -- but DO become unique if sample IDs are attached.'
)
@option(
    '--compress-fastq', 'compress_fastq',
    is_flag=True, default=True,
    help='Specify whether to compress the fastq output files. Currently, only gzip is supported.'
)
def main(
        reads_input: Path,
        out_dir: Path,
        strain_subset_path: Path,
        aligner: str,
        min_read_len: int,
        frac_identity_threshold: float,
        error_threshold: float,
        output_filename: Optional[str],
        attach_sample_ids: bool,
        compress_fastq: bool
):
    """
    Perform filtering on a timeseries dataset, specified by a standard CSV-formatted input index.
    """
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli.filter_timeseries")
    logger.info(f"Performing filtering to timeseries dataset `{reads_input}`.")

    from chronostrain.config import cfg
    from .base import Filter, create_aligner
    from chronostrain.model.io import ReadType
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

    # ============ Prepare output files/directories.
    if output_filename is None or len(output_filename) == 0:
        target_csv_path = out_dir / f"filtered_{reads_input.name}"
    else:
        target_csv_path = out_dir / output_filename

    logger.info(f"Target index file: {target_csv_path}")
    target_csv_path.parent.mkdir(exist_ok=True, parents=True)
    with open(target_csv_path, 'w') as _:
        # Clear the file (Will append in a for loop).
        pass

    # =========== Parse reads.
    filter = Filter(
        db=db,
        strain_collection=strain_collection,
        min_read_len=min_read_len,
        frac_identity_threshold=frac_identity_threshold,
        error_threshold=error_threshold
    )

    if target_csv_path.suffix == '.tsv':
        delim = '\t'
    else:
        delim = ','

    for t, sample_name, read_depth, read_path, read_type_str, qual_fmt in load_from_csv(reads_input, logger=logger):
        read_type = ReadType.parse_from_str(read_type_str)
        logger.info(f"Applying filter to timepoint {t}, {str(read_path)}")

        aligner_obj = create_aligner(aligner, read_type, strain_collection.multifasta_file)
        if attach_sample_ids:
            out_file = f"filtered_{remove_suffixes(read_path).name}-{sample_name}.fastq"
        else:
            out_file = f"filtered_{remove_suffixes(read_path).name}.fastq"

        if compress_fastq:
            out_file = out_file + ".gz"

        filter.apply(read_path, out_dir / out_file, read_type, aligner_obj, quality_format=qual_fmt)
        with open(target_csv_path, 'a') as target_csv:
            # Append to target CSV file.
            writer = csv.writer(target_csv, delimiter=delim, quotechar='\"', quoting=csv.QUOTE_ALL)
            writer.writerow([t, sample_name, read_depth, str(out_file), read_type_str, qual_fmt])

    logger.info("Finished filtering.")


def remove_suffixes(p: Path) -> Path:
    while re.search(r'(\.zip)|(\.gz)|(\.bz2)|(\.fastq)|(\.fq)|(\.fasta)', p.suffix) is not None:
        p = p.with_suffix('')
    return p


def load_from_csv(
        csv_path: Path,
        logger: Logger
) -> List[Tuple[float, str, int, Path, str, str]]:
    time_points: List[Tuple[float, str, int, Path, str, str]] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required file `{str(csv_path)}`")

    logger.debug("Parsing time-series reads from {}".format(csv_path))
    if csv_path.suffix == 'tsv':
        delim = '\t'
    else:
        delim = ','
    with open(csv_path, "r") as f:
        input_specs = csv.reader(f, delimiter=delim, quotechar='"')
        for row in input_specs:
            t = float(row[0])
            sample_name = row[1]
            num_reads = int(row[2])
            read_path = Path(row[3])
            read_type = row[4]
            qual_fmt = row[5]

            if not read_path.is_absolute():
                # assume that the path is relative to the CSV file.
                read_path = csv_path.parent / read_path
            if not read_path.exists():
                raise FileNotFoundError(
                    "The input specification `{}` pointed to `{}`, which does not exist.".format(
                        str(csv_path),
                        read_path
                    ))

            time_points.append((t, sample_name, num_reads, read_path, read_type, qual_fmt))

    time_points = sorted(time_points, key=lambda x: x[0])
    return time_points


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    main_logger = create_logger("chronostrain.MAIN")
    try:
        main()
    except Exception as e:
        main_logger.exception(e)
        exit(1)
