import argparse
import glob
from pathlib import Path
from typing import List, Tuple

from chronostrain.database import StrainDatabase
from chronostrain.model import SequenceRead
from chronostrain.util.alignments import multiple, pairwise
from chronostrain.util.alignments.sam import SamFile
from chronostrain.util.flopp import preprocess
from chronostrain.util.external import run_glopp
from chronostrain.config import cfg, create_logger
logger = create_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create multiple alignment of marker genes to read fragments."
    )

    # Input specification.
    parser.add_argument('-r', '--reads', required=True, type=str,
                        help='<Required> The path (or glob pattern, e.g. `*.fastq`) to fastq files '
                             'containing all reads.')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The target output directory for glopp.')
    parser.add_argument('-p', '--ploidy', required=False, type=int, default=1000,
                        help='<Required> The target ploidy to use for glopp (should be an upper bound to the '
                             'true number).')

    parser.add_argument('-t', '--threads', required=False, type=int,
                        default=cfg.model_cfg.num_cores,
                        help='<Optional> The number of threads to use.')
    return parser.parse_args()


def main():
    args = parse_args()
    db = cfg.database_cfg.get_database()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    run_glopp(
        sam_path=sam_path,
        vcf_path=vcf_path,
        output_dir=out_dir,
        ploidy=ploidy_upper_bound
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        raise
