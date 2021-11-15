import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Iterator, Tuple

from Bio.Seq import Seq
import Bio.SeqIO

from chronostrain import logger, cfg

from chronostrain.database import StrainDatabase
from chronostrain.model.io import TimeSliceReadSource
from chronostrain.util.alignments.sam import SamFile
from chronostrain.util.external.commandline import call_command
from chronostrain.util.alignments.pairwise import parse_alignments, BwaAligner, BowtieAligner

from helpers import parse_input_spec


def file_base_name(file_path: Path) -> str:
    """
    Convert a reference fasta path to the "base name" (typically the accession + marker name).
    e.g. "/data/CP007799.1-16S.fq" -> "CP007799.1-16S"
    """
    return file_path.with_suffix('').name


def filter_on_read_quality(phred_quality: np.ndarray, error_threshold: float = 10):
    num_expected_errors = np.sum(
        np.power(10, -0.1 * phred_quality)
    )
    return num_expected_errors < error_threshold


def filter_on_match_identity(percent_identity: float, identity_threshold=0.9):
    """
    Applies a filtering criteria for reads that continue in the pipeline.
    Currently a simple threshold on percent identity, likely should be adjusted to maximize downstream sensitivity?
    """
    return percent_identity > identity_threshold


def filter_file(
        db: StrainDatabase,
        sam_files: List[Path],
        result_metadata_path: Path,
        result_fq_path: Path,
        quality_format: str,
        min_read_len: int,
        pct_identity_threshold: float
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
            "REVCOMP",
            "PASSED_FILTER",
            "IS_EDGE_MAPPED",
            "LEN",
            "PCT_ID"
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
            if aln.percent_identity is None:
                raise ValueError(f"Unknown percent identity from alignment of read `{aln.read.id}`")

            passed_filter = (
                not aln.is_edge_mapped
                and len(aln.read) > min_read_len
                and filter_on_match_identity(aln.percent_identity, identity_threshold=pct_identity_threshold)
                and filter_on_read_quality(aln.read.quality)
            )

            # Write to metadata file.
            metadata_csv_writer.writerow(
                [
                    aln.read.id,
                    aln.marker.id,
                    aln.marker_start,
                    aln.marker_end,
                    int(aln.reverse_complemented),
                    int(passed_filter),
                    int(aln.is_edge_mapped),
                    len(aln.read),
                    aln.percent_identity
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
                num_report_alignments=self.db.num_markers(),
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
                    pct_identity_threshold=self.pct_identity_threshold
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


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-r', '--reads_dir', required=True, type=str,
                        help='<Required> Directory containing read files. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        dest="output_dir",
                        help='<Required> The file path to save learned outputs to.')

    parser.add_argument('-q', '--quality_format', required=False, type=str, default='fastq',
                        help='<Optional> The quality format. Should be one of the options implemented in Biopython '
                             '`Bio.SeqIO.QualityIO` module.')
    parser.add_argument('-m', '--min_seed_length', required=True, type=int,
                        help='<Required> The minimal seed length to pass to bwa-mem.')

    parser.add_argument('--input_file', required=False, type=str,
                        default='input_files.csv',
                        help='<Optional> The CSV input file specifier inside reads_dir.')
    parser.add_argument('--min_read_len', required=False, type=int, default=35,
                        help='<Optional> Filters out a read if its length was less than the specified value '
                             '(helps reduce spurious alignments). Ideally, trimmomatic should have taken care '
                             'of this step already!')
    parser.add_argument('--pct_identity_threshold', required=False, type=float,
                        default=0.1,
                        help='<Optional> The percent identity threshold at which to filter reads. Default: 0.1.')
    parser.add_argument('--continue_from_idx', required=False, type=int,
                        default=0,
                        help='<Optional> For debugging purposes, assumes that the first N timepoints have already '
                             'been processed, and resumes the filtering at timepoint index N.')
    parser.add_argument('--num_threads', required=False, type=int, default=1,
                        help='<Optional> Specifies the number of threads. Is passed to underlying alignment tools.')

    return parser.parse_args()


def main():
    logger.info("Filtering script active.")
    args = parse_args()
    db = cfg.database_cfg.get_database()

    # =========== Parse reads.
    read_sources, read_depths, time_points = parse_input_spec(
        Path(args.reads_dir) / args.input_file,
        args.quality_format
    )

    # ============ Perform read filtering.
    logger.info("Performing filter on reads.")
    filt = Filter(
        db=db,
        reference_file_path=db.multifasta_file,
        read_sources=read_sources,
        read_depths=read_depths,
        time_points=time_points,
        output_dir=Path(args.output_dir),
        quality_format=args.quality_format,
        min_read_len=args.min_read_len,
        pct_identity_threshold=args.pct_identity_threshold,
        min_seed_length=args.min_seed_length,
        continue_from_idx=args.continue_from_idx,
        num_threads=args.num_threads
    )
    filt.apply_filter(f"filtered_{args.input_file}")
    logger.info("Finished filtering.")


if __name__ == "__main__":
    main()
