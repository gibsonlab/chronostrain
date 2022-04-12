import csv
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

from chronostrain.database import StrainDatabase
from chronostrain.util.alignments.sam import SamFile
from chronostrain.util.external import call_command, bowtie2, bowtie2_build, bt2_func_constant, bt2_func_log
from chronostrain.util.alignments.pairwise import parse_alignments, BowtieAligner, SequenceReadPairwiseAlignment
from chronostrain.config import cfg, create_logger
from chronostrain.util.sequences import nucleotide_GAP_z4

logger = create_logger("chronostrain.filter")


def remove_suffixes(p: Path) -> Path:
    while len(p.suffix) > 0:
        p = p.with_suffix('')
    return p


class Filter(object):
    def __init__(self,
                 db: StrainDatabase,
                 min_read_len: int,
                 pct_identity_threshold: float,
                 error_threshold: float,
                 min_hit_ratio: float = 0.5,
                 num_threads: int = 1):
        self.db = db

        # Note: Bowtie2 does not have the restriction to uncompress bz2 files, but bwa does.
        if self.db.multifasta_file.suffix == ".bz2":
            call_command("bz2", args=["-dk", self.db.multifasta_file])
            self.reference_path = self.db.multifasta_file.with_suffix('')
        else:
            self.reference_path = self.db.multifasta_file

        self.min_read_len = min_read_len
        self.pct_identity_threshold = pct_identity_threshold
        self.error_threshold = error_threshold
        self.min_hit_ratio = min_hit_ratio
        self.num_threads = num_threads

    def apply(self, read_file: Path, out_path: Path, quality_format: str = 'fastq'):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        aligner_tmp_dir = out_path.parent / "tmp"
        aligner_tmp_dir.mkdir(exist_ok=True)

        metadata_path = out_path.parent / f"{remove_suffixes(read_file).name}.metadata.tsv"
        sam_path = aligner_tmp_dir / f"{remove_suffixes(read_file).name}.sam"

        BowtieAligner(
            reference_path=self.reference_path,
            index_basepath=self.reference_path.parent,
            index_basename=self.reference_path.stem,
            num_threads=cfg.model_cfg.num_cores,
            report_all_alignments=False,
            num_reseeds=22,
            score_min_fn=bt2_func_constant(const=-500),
            score_mismatch_penalty=np.floor(
                [np.log(3) + 4 * np.log(10), 0]
            ).astype(int),
            score_read_gap_penalty=np.floor(
                [0, -cfg.model_cfg.get_float("INSERTION_LL_1")]
            ).astype(int),
            score_ref_gap_penalty=np.floor(
                [0, -cfg.model_cfg.get_float("DELETION_LL_1")]
            ).astype(int)
        ).align(query_path=read_file, output_path=sam_path)
        self._apply_helper(sam_path, metadata_path, out_path, quality_format)

    def _apply_helper(
            self,
            sam_path: Path,
            result_metadata_path: Path,
            result_fq_path: Path,
            quality_format: str
    ):
        """
        Parses a sam file and filters reads using the above criteria.
        Writes the results to a fastq file containing the passing reads and a metadata TSV containing
        informative columns.
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
                "PCT_ID_ADJ",
                "START_CLIP",
                "END_CLIP"
            ]
        )

        result_fq = open(result_fq_path, 'w')
        reads_already_passed = set()

        logger.debug(f"Reading: {sam_path.name}")
        for aln in parse_alignments(
                SamFile(sam_path, quality_format),
                self.db,
                min_hit_ratio=self.min_hit_ratio
        ):
            if aln.read.id in reads_already_passed:
                # Read is already included in output file. Don't do anything.
                continue

            # Pass filter if quality is high enough, and entire read is mapped.
            filter_edge_clip = self.filter_on_ungapped_bases(aln)
            percent_identity_adjusted = self.adjusted_match_identity(aln)

            passed_filter = (
                    filter_edge_clip
                    and len(aln.read) > self.min_read_len
                    and percent_identity_adjusted > self.pct_identity_threshold
                    and self.num_expected_errors(aln) < self.error_threshold
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
                    percent_identity_adjusted,
                    aln.soft_clip_start + aln.hard_clip_start,
                    aln.soft_clip_end + aln.hard_clip_end
                ]
            )

            if passed_filter:
                # Add to collection of already added reads.
                reads_already_passed.add(aln.read.id)

                # Write SeqRecord to file.
                record = SeqIO.SeqRecord(
                    Seq(aln.read.nucleotide_content()),
                    id=aln.read.id,
                    description="{}_{}:{}".format(aln.marker.id, aln.marker_start, aln.marker_end)
                )
                record.letter_annotations["phred_quality"] = aln.read.quality
                SeqIO.write(record, result_fq, "fastq")
        logger.debug(f"# passed reads: {len(reads_already_passed)}")
        result_metadata.close()
        result_fq.close()

    def filter_on_ungapped_bases(self, aln: SequenceReadPairwiseAlignment):
        return np.sum(aln.aln_matrix[1] != nucleotide_GAP_z4) / len(aln.read) > self.min_hit_ratio

    def adjusted_match_identity(self, aln: SequenceReadPairwiseAlignment):
        """
        Applies a filtering criteria for reads that continue in the pipeline.
        Currently a simple threshold on percent identity, likely should be adjusted to maximize downstream sensitivity?
        """
        if aln.num_aligned_bases is None:
            raise ValueError(f"Unknown num_aligned_bases from alignment of read `{aln.read.id}`")
        if aln.num_mismatches is None:
            raise ValueError(f"Unknown num_mismatches from alignment of read `{aln.read.id}`")

        n_expected_errors = self.num_expected_errors(aln)
        adjusted_pct_identity = self.clip_between(
            1.0 - ((aln.num_mismatches - n_expected_errors) / (aln.num_aligned_bases - n_expected_errors)),
            lower=0.0,
            upper=1.0,
        )

        return adjusted_pct_identity

    @staticmethod
    def num_expected_errors(aln: SequenceReadPairwiseAlignment):
        read_qual = aln.read.quality
        if aln.reverse_complemented:
            read_qual = read_qual[::-1]

        read_start_clip = aln.soft_clip_start + aln.hard_clip_start
        read_end_clip = aln.hard_clip_end + aln.soft_clip_end
        _slice = slice(read_start_clip, len(read_qual) - read_end_clip)

        marker_aln, read_aln = aln.aln_matrix[0], aln.aln_matrix[1]
        insertion_locs = np.equal(marker_aln, nucleotide_GAP_z4)[read_aln != nucleotide_GAP_z4]
        read_qual = read_qual[_slice][~insertion_locs]

        return np.sum(
            np.power(10, -0.1 * read_qual)
        )

    @staticmethod
    def clip_between(x: float, lower: float, upper: float) -> float:
        return max(min(x, upper), lower)
