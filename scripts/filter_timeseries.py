import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from chronostrain import create_logger, cfg
from helpers.filter import Filter

logger = create_logger("chronostrain.filter_timeseries")


def remove_suffixes(p: Path) -> Path:
    while len(p.suffix) > 0:
        p = p.with_suffix('')
    return p


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-r', '--reads_input', required=True, type=str,
                        help='<Required> Path to the reads input CSV file.')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The file path to save filtered output fastq files.')

    parser.add_argument('-ro', '--reads_output_filename', required=False, type=str,
                        default='',
                        help='<Optional> The filename of the target CSV file. If not specified, will default to'
                             '`filtered_<input_file>`. Parent directory is always out_dir.')
    parser.add_argument('--min_read_len', required=False, type=int, default=35,
                        help='<Optional> Filters out a read if its length was less than the specified value '
                             '(helps reduce spurious alignments). Ideally, trimmomatic should have taken care '
                             'of this step already!')
    parser.add_argument('--pct_identity_threshold', required=False, type=float,
                        default=0.1,
                        help='<Optional> The percent identity threshold at which to filter reads. Default: 0.1.')
    parser.add_argument('--error_threshold', required=False, type=float,
                        default=10.0,
                        help='<Optional> The maximum number of expected errors tolerated in order to pass filter.'
                             'Default: 10.0')
    parser.add_argument('--num_threads', required=False, type=int,
                        default=cfg.model_cfg.num_cores,
                        help='<Optional> Specifies the number of threads. Is passed to underlying alignment tools.')
    return parser.parse_args()


def load_from_csv(csv_path: Path) -> List[Tuple[float, int, Path, str, str]]:
    import csv

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


def main():
    args = parse_args()
    logger.info(f"Performing filtering to timeseries dataset `{args.reads_input}`.")
    db = cfg.database_cfg.get_database()

    # ============ Prepare output files/directories.
    out_dir = Path(args.out_dir)
    if args.reads_output_filename == '':
        target_csv_path = out_dir / f"filtered_{Path(args.reads_input).name}"
    else:
        target_csv_path = out_dir / args.reads_output_filename

    logger.info(f"Target index file: {target_csv_path}")
    target_csv_path.parent.mkdir(exist_ok=True, parents=True)
    with open(target_csv_path, 'w') as _:
        # Clear the file (Will append in a for loop).
        pass

    # =========== Parse reads.
    filter = Filter(
        db=db,
        min_read_len=args.min_read_len,
        pct_identity_threshold=args.pct_identity_threshold,
        error_threshold=args.error_threshold,
        num_threads=args.num_threads
    )
    for t, read_depth, read_path, read_type, qual_fmt in load_from_csv(Path(args.reads_input)):
        logger.info(f"Applying filter to timepoint {t}, {str(read_path)}")
        out_path = out_dir / f"filtered_{remove_suffixes(read_path).name}.fastq"
        filter.apply(read_path, out_path, quality_format=qual_fmt)
        with open(target_csv_path, 'a') as target_csv:
            # Append to target CSV file.
            writer = csv.writer(target_csv, delimiter=',', quotechar='\"', quoting=csv.QUOTE_ALL)
            writer.writerow([t, read_depth, str(out_path), read_type, qual_fmt])

    logger.info("Finished filtering.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        exit(1)
