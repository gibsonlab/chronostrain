import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
import Bio.SeqIO

from chronostrain import create_logger, cfg
logger = create_logger("chronostrain.filter")

from chronostrain.database import StrainDatabase
from chronostrain.util.alignments.sam import SamFile
from chronostrain.util.external.commandline import call_command
from chronostrain.util.alignments.pairwise import parse_alignments, BwaAligner, BowtieAligner, \
    SequenceReadPairwiseAlignment


def remove_suffixes(p: Path) -> Path:
    while len(p.suffix) > 0:
        p = p.with_suffix('')
    return p


def num_expected_errors(aln: SequenceReadPairwiseAlignment):
    return np.sum(
        np.power(10, -0.1 * aln.read.quality)
    )


def clip_between(x: float, lower: float, upper: float) -> float:
    return max(min(x, upper), lower)


def adjusted_match_identity(aln: SequenceReadPairwiseAlignment):
    """
    Applies a filtering criteria for reads that continue in the pipeline.
    Currently a simple threshold on percent identity, likely should be adjusted to maximize downstream sensitivity?
    """
    if aln.num_aligned_bases is None:
        raise ValueError(f"Unknown num_aligned_bases from alignment of read `{aln.read.id}`")
    if aln.num_mismatches is None:
        raise ValueError(f"Unknown num_mismatches from alignment of read `{aln.read.id}`")

    n_expected_errors = num_expected_errors(aln)
    adjusted_pct_identity = clip_between(
        1.0 - ((aln.num_mismatches - n_expected_errors) / (aln.num_aligned_bases - n_expected_errors)),
        lower=0.0,
        upper=1.0,
    )

    return adjusted_pct_identity


def filter_on_edge_clip(aln: SequenceReadPairwiseAlignment, clip_fraction: float = 0.5):
    if aln.is_edge_mapped:
        # Fail if start and end are both soft clipped.
        if (aln.soft_clip_start > 0 or aln.hard_clip_start > 0) and (aln.soft_clip_end > 0 or aln.hard_clip_end > 0):
            return False

        if aln.soft_clip_start > 0 or aln.hard_clip_start > 0:
            return (
                    (aln.soft_clip_start / len(aln.read)) < clip_fraction
                    and (aln.hard_clip_start / len(aln.read)) < clip_fraction
            )

        if aln.soft_clip_end > 0 or aln.hard_clip_end > 0:
            return (
                    (aln.soft_clip_end / len(aln.read)) < clip_fraction
                    and (aln.hard_clip_end / len(aln.read)) < clip_fraction
            )
    else:
        return True


def filter_file(
        db: StrainDatabase,
        sam_path: Path,
        result_metadata_path: Path,
        result_fq_path: Path,
        quality_format: str,
        min_read_len: int,
        pct_identity_threshold: float,
        error_threshold: float
):
    """
    Parses a sam file and filters reads using the above criteria.
    Writes the results to a fastq file containing the passing reads and a metadata TSV containing columns:
        Read Name    Percent Identity    Passes Filter?
    """

    result_metadata = open(result_metadata_path, 'w')
    metadata_csv_writer = csv.writer(result_metadata, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    metadata_csv_writer.writerow(
        [
            "READ",
            "MARKER",
            "MARKER_START",
            "MARKER_END",
            "PASSED_FILTER",
            "REVCOMP",
            "IS_EDGE_MAPPED",
            "READ_LEN",
            "N_MISMATCHES",
            "PCT_ID_ADJ"
        ]
    )

    result_fq = open(result_fq_path, 'w')
    reads_already_passed = set()

    logger.info(f"Reading: {sam_path.name}")
    for aln in parse_alignments(
            SamFile(sam_path, quality_format), db
    ):
        if aln.read.id in reads_already_passed:
            # Read is already included in output file. Don't do anything.
            continue

        # Pass filter if quality is high enough, and entire read is mapped.
        filter_edge_clip = filter_on_edge_clip(aln, clip_fraction=0.25)
        percent_identity_adjusted = adjusted_match_identity(aln)

        passed_filter = (
            filter_edge_clip
            and len(aln.read) > min_read_len
            and percent_identity_adjusted > pct_identity_threshold
            and num_expected_errors(aln) < error_threshold
        )

        # Write to metadata file.
        metadata_csv_writer.writerow(
            [
                aln.read.id,
                aln.marker.id,
                aln.marker_start,
                aln.marker_end,
                int(passed_filter),
                int(aln.reverse_complemented),
                int(aln.is_edge_mapped),
                len(aln.read),
                aln.num_mismatches,
                percent_identity_adjusted
            ]
        )

        if passed_filter:
            # Add to collection of already added reads.
            reads_already_passed.add(aln.read.id)

            # Write SeqRecord to file.
            record = Bio.SeqIO.SeqRecord(
                Seq(aln.read.nucleotide_content()),
                id=aln.read.id,
                description="{}_{}:{}".format(aln.marker.id, aln.marker_start, aln.marker_end)
            )
            record.letter_annotations["phred_quality"] = aln.read.quality
            Bio.SeqIO.write(record, result_fq, "fastq")
    logger.info(f"# passed reads: {len(reads_already_passed)}")
    result_metadata.close()
    result_fq.close()


class Filter:
    def __init__(self,
                 db: StrainDatabase,
                 reference_file_path: Path,
                 time_points: List[Tuple[float, int, Path, str, str]],
                 output_dir: Path,
                 min_seed_length: int,
                 min_read_len: int,
                 pct_identity_threshold: float,
                 error_threshold: float,
                 num_threads: int = 1):
        logger.debug("Reference path: {}".format(reference_file_path))

        self.db = db

        # Note: Bowtie2 does not have the restriction to uncompress bz2 files, but bwa does.
        if reference_file_path.suffix == ".bz2":
            call_command("bz2", args=["-dk", reference_file_path])
            self.reference_path = reference_file_path.with_suffix('')
        else:
            self.reference_path = reference_file_path

        self.time_points = time_points
        self.min_seed_length = min_seed_length

        self.output_dir = output_dir
        self.min_read_len = min_read_len
        self.pct_identity_threshold = pct_identity_threshold
        self.num_threads = num_threads
        self.error_threshold = error_threshold

    def apply_filter(self, destination_csv: str):
        """
        :return: A list of paths to the resulting filtered read files.
        """
        if cfg.external_tools_cfg.pairwise_align_cmd == "bwa":
            aligner = BwaAligner(
                reference_path=self.reference_path,
                min_seed_len=8,
                num_threads=cfg.model_cfg.num_cores,
                report_all_alignments=True
            )
        elif cfg.external_tools_cfg.pairwise_align_cmd == "bowtie2":
            aligner = BowtieAligner(
                reference_path=self.reference_path,
                index_basepath=self.reference_path.parent,
                index_basename="markers",
                num_threads=cfg.model_cfg.num_cores
            )
        else:
            raise NotImplementedError(
                f"Alignment command `{cfg.external_tools_cfg.pairwise_align_cmd}` not currently supported."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        aligner_tmp_dir = self.output_dir / "tmp"
        aligner_tmp_dir.mkdir(parents=True, exist_ok=True)

        csv_rows: List[Tuple[float, int, Path, str, str]] = []

        for time_point, n_reads, filepath, read_type, qual_fmt in self.time_points:
            logger.info(f"Applying filter to (t={time_point}) {str(filepath)}")
            result_metadata_path = self.output_dir / 'metadata_{}.tsv'.format(time_point)
            result_fq_path = self.output_dir / f"filtered_{remove_suffixes(filepath).name}"

            sam_path = aligner_tmp_dir / remove_suffixes(filepath).name

            aligner.align(query_path=filepath, output_path=sam_path)

            filter_file(
                db=self.db,
                sam_path=sam_path,
                result_metadata_path=result_metadata_path,
                result_fq_path=result_fq_path,
                quality_format=qual_fmt,
                min_read_len=self.min_read_len,
                pct_identity_threshold=self.pct_identity_threshold,
                error_threshold=self.error_threshold
            )
            logger.info("Timepoint {t}, filtered reads file: {f}".format(
                t=time_point, f=result_fq_path
            ))

            csv_rows.append((time_point, n_reads, result_fq_path, read_type, qual_fmt))
        save_input_csv(csv_rows, self.output_dir / destination_csv)


def save_input_csv(csv_rows: List[Tuple[float, int, Path, str, str]],
                   out_path: Path):
    """
    Generates the target input.csv file pointing to the proper sources (of the filtered reads).
    """
    with open(out_path, "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='\"', quoting=csv.QUOTE_ALL)
        for time_point, read_depth, filtered_file, read_type, qual_fmt in csv_rows:
            writer.writerow([
                time_point,
                read_depth,
                str(filtered_file),
                read_type,
                qual_fmt
            ])


def get_canonical_multifasta(db: StrainDatabase) -> Path:
    out_path = db.multifasta_file.with_stem(f"{db.multifasta_file.stem}_canonical")

    SeqIO.write(
        [marker.to_seqrecord() for marker in db.all_canonical_markers()],
        out_path,
        "fasta"
    )

    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-r', '--reads_input', required=True, type=str,
                        help='<Required> Path to the reads input CSV file. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        dest="output_dir",
                        help='<Required> The file path to save learned outputs to.')

    parser.add_argument('-m', '--min_seed_length', required=True, type=int,
                        help='<Required> The minimal seed length to pass to bwa-mem.')

    parser.add_argument('--min_read_len', required=False, type=int, default=35,
                        help='<Optional> Filters out a read if its length was less than the specified value '
                             '(helps reduce spurious alignments). Ideally, trimmomatic should have taken care '
                             'of this step already!')
    parser.add_argument('--pct_identity_threshold', required=False, type=float,
                        default=0.1,
                        help='<Optional> The percent identity threshold at which to filter reads. Default: 0.1.')
    parser.add_argument('--phred_error_threshold', required=False, type=float,
                        default=10.0,
                        help='<Optional> The maximum number of expected errors tolerated in order to pass filter.'
                             'Default: 10.0')
    parser.add_argument('--num_threads', required=False, type=int, default=1,
                        help='<Optional> Specifies the number of threads. Is passed to underlying alignment tools.')
    parser.add_argument('--canonical_only', action='store_true',
                        help='If flag is enabled, then alignment filtering is only done with respect to canonical'
                             'markers. (Useful for simulated data, where non-canonical markers might have been'
                             'used to sample the reads, and one does not want to include the ground truth.)')

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
    logger.info("Filtering script active.")
    args = parse_args()
    db = cfg.database_cfg.get_database()

    # =========== Parse reads.
    time_points = load_from_csv(
        Path(args.reads_input)
    )

    if args.canonical_only:
        reference_path = get_canonical_multifasta(db)
    else:
        reference_path = db.multifasta_file

    # ============ Perform read filtering.
    target_file = f"filtered_{Path(args.reads_input).name}"
    logger.info(f"Target index file: {target_file}")

    logger.info("Performing filter on reads.")
    filt = Filter(
        db=db,
        reference_file_path=reference_path,
        time_points=time_points,
        output_dir=Path(args.output_dir),
        min_read_len=args.min_read_len,
        pct_identity_threshold=args.pct_identity_threshold,
        min_seed_length=args.min_seed_length,
        num_threads=args.num_threads,
        error_threshold=args.phred_error_threshold
    )

    filt.apply_filter(target_file)
    logger.info("Finished filtering.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        exit(1)
