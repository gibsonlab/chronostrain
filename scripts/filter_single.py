import argparse
from pathlib import Path
import numpy as np

from chronostrain import create_logger, cfg
from chronostrain.database import StrainDatabase
from chronostrain.model.io import ReadType, parse_read_type
from chronostrain.util.alignments.pairwise import AbstractPairwiseAligner, BwaAligner, BowtieAligner
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
    parser.add_argument('--read_type', required=True, type=str)

    # Optional params.
    parser.add_argument('--aligner', required=False, type=str, default='bwa',
                        help='<Optional> Specify the aligner to use. (Options: bwa, bowtie2) (Default: bwa)')
    parser.add_argument('--min_read_len', required=False, type=int, default=35,
                        help='<Optional> Filters out a read if its length was less than the specified value '
                             '(helps reduce spurious alignments). Ideally, trimmomatic should have taken care '
                             'of this step already!')
    parser.add_argument('--pct_identity_threshold', required=False, type=float,
                        default=0.1,
                        help='<Optional> The percent identity threshold at which to filter reads. Default: 0.1.')
    parser.add_argument('--error_threshold', required=False, type=float,
                        default=0.05,
                        help='<Optional> The number of expected errors tolerated in order to pass filter, '
                             'expressed as a ratio to the length of the read..'
                             'Default: 0.05')
    parser.add_argument('--num_threads', required=False, type=int,
                        default=cfg.model_cfg.num_cores,
                        help='<Optional> Specifies the number of threads. Is passed to underlying alignment tools.')
    return parser.parse_args()


def create_aligner(aligner_type: str, read_type: ReadType, db: StrainDatabase) -> AbstractPairwiseAligner:
    if read_type == ReadType.PAIRED_END_1:
        insertion_ll = cfg.model_cfg.get_float("INSERTION_LL_1")
        deletion_ll = cfg.model_cfg.get_float("DELETION_LL_1")
    elif read_type == ReadType.PAIRED_END_2:
        insertion_ll = cfg.model_cfg.get_float("INSERTION_LL_2")
        deletion_ll = cfg.model_cfg.get_float("DELETION_LL_2")
    elif read_type == ReadType.SINGLE_END:
        insertion_ll = cfg.model_cfg.get_float("INSERTION_LL")
        deletion_ll = cfg.model_cfg.get_float("DELETION_LL")
    else:
        raise ValueError(f"Unrecognized read type `{read_type}`.")

    if aligner_type == 'bwa':
        return BwaAligner(
            reference_path=db.multifasta_file,
            min_seed_len=15,
            reseed_ratio=0.5,  # default; smaller = slower but more alignments.
            bandwidth=10,
            num_threads=cfg.model_cfg.num_cores,
            report_all_alignments=False,
            match_score=2,  # log likelihood ratio log_2(4p)
            mismatch_penalty=5,  # Assume quality score of 20, log likelihood ratio log_2(4 * error * <3/4>)
            off_diag_dropoff=100,  # default
            gap_open_penalty=(0, 0),
            gap_extend_penalty=(
                int(-deletion_ll / np.log(2)),
                int(-insertion_ll / np.log(2))
            ),
            clip_penalty=5,
            score_threshold=50,
            bwa_command='bwa-mem2'
        )
    else:
        from chronostrain.util.external import bt2_func_constant
        return BowtieAligner(
            reference_path=db.multifasta_file,
            index_basepath=db.multifasta_file.parent,
            index_basename=db.multifasta_file.stem,
            num_threads=cfg.model_cfg.num_cores,
            report_all_alignments=False,
            seed_length=22,  # -L 22
            seed_num_mismatches=0,  # -N 0
            seed_extend_failures=5,  # -D 5
            num_reseeds=1,  # -R 1
            score_min_fn=bt2_func_constant(const=50),
            score_match_bonus=2,
            score_mismatch_penalty=np.floor(
                [5, 5]
            ).astype(int),
            score_read_gap_penalty=np.floor(
                [0, int(-deletion_ll / np.log(2))]
            ).astype(int),
            score_ref_gap_penalty=np.floor(
                [0, int(-insertion_ll / np.log(2))]
            ).astype(int)
        )


def main():
    args = parse_args()
    logger.info(f"Applying filter to `{args.in_path}`")

    db = cfg.database_cfg.get_database()

    # =========== Parse reads.
    filter = Filter(
        db=db,
        min_read_len=args.min_read_len,
        frac_identity_threshold=args.frac_identity_threshold,
        error_threshold=args.error_threshold,
        num_threads=args.num_threads
    )

    read_type = parse_read_type(args.read_type)
    aligner = create_aligner(args.aligner, read_type, db)
    filter.apply(
        Path(args.in_path),
        Path(args.out_path),
        aligner=aligner,
        read_type=read_type,
        quality_format=args.quality_format
    )

    logger.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        exit(1)
