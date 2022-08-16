import click
import re
import csv
from logging import Logger
from pathlib import Path
from typing import List, Tuple, Optional

from ..base import option


@click.command()
@click.pass_context
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
    '--output-filename', '-f', 'reads_output_filename',
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
        reads_input: Path,
        out_dir: Path,
        min_read_len: int,
        frac_identity_threshold: float,
        error_threshold: float,
        output_filename: Optional[str]
):
    """
    Perform filtering on a timeseries dataset, specified by a standard CSV-formatted input index.
    """
    ctx.ensure_object(Logger)
    logger = ctx.obj
    logger.info(f"Performing filtering to timeseries dataset `{reads_input}`.")

    from chronostrain.config import cfg
    from .base import Filter
    db = cfg.database_cfg.get_database()

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
        min_read_len=min_read_len,
        frac_identity_threshold=frac_identity_threshold,
        error_threshold=error_threshold,
        num_threads=cfg.model_cfg.num_cores
    )

    for t, read_depth, read_path, read_type, qual_fmt in load_from_csv(reads_input):
        logger.info(f"Applying filter to timepoint {t}, {str(read_path)}")
        out_path = out_dir / f"filtered_{remove_suffixes(read_path).name}.fastq"
        filter.apply(read_path, out_path, read_type, quality_format=qual_fmt)
        with open(target_csv_path, 'a') as target_csv:
            # Append to target CSV file.
            writer = csv.writer(target_csv, delimiter=',', quotechar='\"', quoting=csv.QUOTE_ALL)
            writer.writerow([t, read_depth, str(out_path), read_type, qual_fmt])

    logger.info("Finished filtering.")


def remove_suffixes(p: Path) -> Path:
    while re.search(r'(\.zip)|(\.gz)|(\.bz2)|(\.fastq)|(\.fq)|(\.fasta)', p.suffix) is not None:
        p = p.with_suffix('')
    return p


def load_from_csv(csv_path: Path, logger: Logger) -> List[Tuple[float, int, Path, str, str]]:
    time_points: List[Tuple[float, int, Path, str, str]] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required file `{str(csv_path)}`")

    logger.debug("Parsing time-series reads from {}".format(csv_path))
    with open(csv_path, "r") as f:
        input_specs = csv.reader(f, delimiter=',', quotechar='"')
        for row in input_specs:
            t = float(row[0])
            num_reads = int(row[1])
            read_path = Path(row[2])
            read_type = row[3]
            qual_fmt = row[4]

            if not read_path.exists():
                raise FileNotFoundError(
                    "The input specification `{}` pointed to `{}`, which does not exist.".format(
                        str(csv_path),
                        read_path
                    ))

            time_points.append((t, num_reads, read_path, read_type, qual_fmt))

    time_points = sorted(time_points, key=lambda x: x[0])
    return time_points


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.filter")
    try:
        main(obj=logger)
    except BaseException as e:
        logger.exception(e)
        exit(1)
