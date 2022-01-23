import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Iterator, Tuple, Dict

from Bio import SeqIO
from Bio.Seq import Seq
import Bio.SeqIO

from chronostrain import create_logger, cfg
logger = create_logger("chronostrain.filter")

from chronostrain.database import StrainDatabase
from chronostrain.model.io import TimeSliceReadSource
from chronostrain.util.alignments.sam import SamFile
from chronostrain.util.external.commandline import call_command
from chronostrain.util.alignments.pairwise import parse_alignments, BwaAligner, BowtieAligner, \
    SequenceReadPairwiseAlignment


def file_base_name(file_path: Path) -> str:
    """
    Convert a reference fasta path to the "base name" (typically the accession + marker name).
    e.g. "/data/CP007799.1-16S.fq" -> "CP007799.1-16S"
    """
    return file_path.with_suffix('').name


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
        sam_files: List[Path],
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

    for sam_file_path in sam_files:
        logger.info(f"Reading: {sam_file_path.name}")
        for aln in parse_alignments(
                SamFile(sam_file_path, quality_format), db
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
                 read_sources: List[TimeSliceReadSource],
                 read_depths: List[int],
                 time_points: List[float],
                 output_dir: Path,
                 quality_format: str,
                 min_seed_length: int,
                 min_read_len: int,
                 pct_identity_threshold: float,
                 error_threshold: float,
                 continue_from_idx: int = 0,
                 num_threads: int = 1):
        logger.debug("Reference path: {}".format(reference_file_path))

        self.db = db

        # Note: Bowtie2 does not have the restriction to uncompress bz2 files, but bwa does.
        if reference_file_path.suffix == ".bz2":
            call_command("bz2", args=["-dk", reference_file_path])
            self.reference_path = reference_file_path.with_suffix('')
        else:
            self.reference_path = reference_file_path

        self.read_sources = read_sources
        self.read_depths = read_depths
        self.time_points = time_points
        self.min_seed_length = min_seed_length

        self.output_dir = output_dir
        self.quality_format = quality_format
        self.min_read_len = min_read_len
        self.pct_identity_threshold = pct_identity_threshold
        self.continue_from_idx = continue_from_idx
        self.num_threads = num_threads
        self.error_threshold = error_threshold

    def time_point_specs(self) -> Iterator[Tuple[TimeSliceReadSource, int, float]]:
        yield from zip(self.read_sources, self.read_depths, self.time_points)

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
        resulting_files: List[Path] = []
        for t_idx, (src, read_depth, time_point) in enumerate(self.time_point_specs()):
            result_metadata_path = self.output_dir / 'metadata_{}.tsv'.format(time_point)
            result_fq_path = self.output_dir / "reads_{}.fq".format(time_point)

            if t_idx >= self.continue_from_idx:
                sam_paths_t = []
                for read_path in src.paths:
                    base_path = read_path.parent
                    aligner_tmp_dir = base_path / "tmp"
                    aligner_tmp_dir.mkdir(parents=True, exist_ok=True)

                    sam_path = aligner_tmp_dir / "{}.sam".format(
                        file_base_name(read_path)
                    )

                    aligner.align(query_path=read_path, output_path=sam_path)
                    sam_paths_t.append(sam_path)

                logger.debug("(t = {}) Reading SAM files {}".format(
                    time_point,
                    ",".join(str(p) for p in sam_paths_t)
                ))

                filter_file(
                    db=self.db,
                    sam_files=sam_paths_t,
                    result_metadata_path=result_metadata_path,
                    result_fq_path=result_fq_path,
                    quality_format=self.quality_format,
                    min_read_len=self.min_read_len,
                    pct_identity_threshold=self.pct_identity_threshold,
                    error_threshold=self.error_threshold
                )
                logger.info("Timepoint {t}, filtered reads file: {f}".format(
                    t=time_point, f=result_fq_path
                ))
            else:
                pass

            resulting_files.append(result_fq_path)

        save_input_csv(self.time_points, self.read_depths, self.output_dir / destination_csv, resulting_files)


def save_input_csv(time_points: List[float],
                   read_depths: List[int],
                   out_path: Path,
                   filtered_files: List[Path]):
    """
    Generates the target input.csv file pointing to the proper sources (of the filtered reads).
    """
    with open(out_path, "w") as f:
        writer = csv.writer(f, delimiter=',', quotechar='\"', quoting=csv.QUOTE_ALL)
        for time_point, read_depth, filtered_file in zip(time_points, read_depths, filtered_files):
            writer.writerow([
                time_point,
                read_depth,
                str(filtered_file)
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

    parser.add_argument('-q', '--quality_format', required=False, type=str, default='fastq',
                        help='<Optional> The quality format. Should be one of the options implemented in Biopython '
                             '`Bio.SeqIO.QualityIO` module.')
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
    parser.add_argument('--continue_from_idx', required=False, type=int,
                        default=0,
                        help='<Optional> For debugging purposes, assumes that the first N timepoints have already '
                             'been processed, and resumes the filtering at timepoint index N.')
    parser.add_argument('--num_threads', required=False, type=int, default=1,
                        help='<Optional> Specifies the number of threads. Is passed to underlying alignment tools.')
    parser.add_argument('--canonical_only', action='store_true',
                        help='If flag is enabled, then alignment filtering is only done with respect to canonical'
                             'markers. (Useful for simulated data, where non-canonical markers might have been'
                             'used to sample the reads, and one does not want to include the ground truth.)')

    return parser.parse_args()


def load_from_csv(csv_path: Path, quality_format: str) -> 'TimeSeriesReads':
    import csv
    time_points_to_reads: Dict[float, List[Tuple[int, Path]]] = {}
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required file `{str(csv_path)}`")

    logger.debug("Parsing time-series reads from {}".format(csv_path))
    with open(csv_path, "r") as f:
        input_specs = csv.reader(f, delimiter=',', quotechar='"')
        for row in input_specs:
            time_point = float(row[0])
            num_reads = int(row[1])
            read_path = Path(row[2])

            if not read_path.exists():
                raise FileNotFoundError(
                    "The input specification `{}` pointed to `{}`, which does not exist.".format(
                        str(csv_path),
                        read_path
                    ))

            if time_point not in time_points_to_reads:
                time_points_to_reads[time_point] = []

            time_points_to_reads[time_point].append((num_reads, read_path))

    time_points = sorted(time_points_to_reads.keys(), reverse=False)
    logger.info("Found timepoints: {}".format(time_points))

    read_depths = [
        sum([
            n_reads for n_reads, _ in time_points_to_reads[t]
        ])
        for t in time_points
    ]

    time_slice_sources = [
        TimeSliceReadSource([
            read_path for _, read_path in time_points_to_reads[t]
        ], quality_format)
        for t in time_points
    ]

    return time_slice_sources, read_depths, time_points


def main():
    logger.info("Filtering script active.")
    args = parse_args()
    db = cfg.database_cfg.get_database()

    # =========== Parse reads.
    read_sources, read_depths, time_points = load_from_csv(
        Path(args.reads_input),
        quality_format=args.quality_format
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
        read_sources=read_sources,
        read_depths=read_depths,
        time_points=time_points,
        output_dir=Path(args.output_dir),
        quality_format=args.quality_format,
        min_read_len=args.min_read_len,
        pct_identity_threshold=args.pct_identity_threshold,
        min_seed_length=args.min_seed_length,
        continue_from_idx=args.continue_from_idx,
        num_threads=args.num_threads,
        error_threshold=args.phred_error_threshold
    )

    filt.apply_filter(target_file)
    logger.info("Finished filtering.")


if __name__ == "__main__":
    main()
