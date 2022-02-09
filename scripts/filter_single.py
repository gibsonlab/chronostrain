import argparse
from pathlib import Path

from chronostrain import create_logger, cfg
from helpers.filter import Filter

logger = create_logger("chronostrain.filter_single")


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-i', '--in_path', required=True, type=str,
                        help='<Required> The input file path.')
    parser.add_argument('-o', '--out_path', required=True, type=str,
                        help='<Required> The output file path (fastq format).')
    parser.add_argument('-q', '--quality_format', required=False, type=str,
                        default='fastq',
                        help='<Optional> The quality format of the input file. '
                             'Must be parsable by Bio.SeqIO. '
                             'Default: `fastq`')

    # Optional params.
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


def main():
    args = parse_args()
    logger.info(f"Applying filter to `{args.in_path}`")

    db = cfg.database_cfg.get_database()

    # =========== Parse reads.
    filter = Filter(
        db=db,
        min_read_len=args.min_read_len,
        pct_identity_threshold=args.pct_identity_threshold,
        error_threshold=args.error_threshold,
        num_threads=args.num_threads
    )

    filter.apply(
        Path(args.in_path),
        Path(args.out_path),
        quality_format=args.quality_format
    )

    logger.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        exit(1)
