import csv
from pathlib import Path
import re

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

from chronostrain.database import StrainDatabase
from chronostrain.model.io import ReadType
from chronostrain.util.alignments.sam import SamFile
from chronostrain.util.external import call_command
from chronostrain.util.alignments.pairwise import *
from chronostrain.util.sequences import nucleotide_GAP_z4

from chronostrain.config import cfg
from chronostrain.logging import create_logger
logger = create_logger(__name__)


def remove_suffixes(p: Path) -> Path:
    while re.search(r'(\.zip)|(\.gz)|(\.bz2)|(\.fastq)|(\.fq)|(\.fasta)', p.suffix) is not None:
        p = p.with_suffix('')
    return p


class Filter(object):
    def __init__(self,
                 db: StrainDatabase,
                 min_read_len: int,
                 frac_identity_threshold: float,
                 error_threshold: float = 1.0,
                 min_hit_ratio: float = 0.5):
        self.db = db

        # Note: Bowtie2 does not have the restriction to uncompress bz2 files, but bwa does.
        if self.db.multifasta_file.suffix == ".bz2":
            call_command("bz2", args=["-dk", self.db.multifasta_file])
            self.reference_path = self.db.multifasta_file.with_suffix('')
        else:
            self.reference_path = self.db.multifasta_file

        self.min_read_len = min_read_len
        self.frac_identity_threshold = frac_identity_threshold
        self.error_threshold = error_threshold
        self.min_hit_ratio = min_hit_ratio

    def apply(self, read_file: Path, out_path: Path, read_type: ReadType, aligner: AbstractPairwiseAligner, quality_format: str = 'fastq'):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        aligner_tmp_dir = out_path.parent / "tmp"
        aligner_tmp_dir.mkdir(exist_ok=True)

        metadata_path = out_path.parent / f"{remove_suffixes(read_file).name}.metadata.tsv"
        sam_path = aligner_tmp_dir / f"{remove_suffixes(read_file).name}.sam"
        aligner.align(
            query_path=read_file,
            output_path=sam_path,
            read_type=read_type
        )
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
                "PCT_ID",
                "EXP_ERRORS",
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
                reattach_clipped_bases=True,
                min_hit_ratio=self.min_hit_ratio
        ):
            if aln.read.id in reads_already_passed:
                # Read is already included in output file. Don't do anything.
                continue

            # Pass filter if quality is high enough, and
            # enough of the bases are mapped (if read maps to the edge of a marker).
            filter_edge_clip = self.filter_on_ungapped_bases(aln)
            frac_identity = aln.num_matches / len(aln.read)
            aln_frac_identity = aln.num_matches / aln.aln_matrix.shape[1]
            n_exp_errors = self.num_expected_errors(aln)

            passed_filter = (
                not aln.is_edge_mapped
                and filter_edge_clip
                and len(aln.read) > self.min_read_len
                and frac_identity > self.frac_identity_threshold
                and n_exp_errors < (self.error_threshold * len(aln.read))
            ) or (
                aln.is_edge_mapped
                and filter_edge_clip
                and len(aln.read) > self.min_read_len
                and aln_frac_identity > self.frac_identity_threshold
                and n_exp_errors < (self.error_threshold * len(aln.read))
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
                    frac_identity * 100.0,
                    n_exp_errors,
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
            clip_penalty=0,
            score_threshold=50,
            bwa_command='bwa'
        )
    elif aligner_type == 'bwa-mem2':
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
            clip_penalty=0,
            score_threshold=50,
            bwa_command='bwa-mem2'
        )
    elif aligner_type == 'bowtie2':
        from chronostrain.util.external import bt2_func_constant
        return BowtieAligner(
            reference_path=db.multifasta_file,
            index_basepath=db.multifasta_file.parent,
            index_basename=db.multifasta_file.stem,
            num_threads=cfg.model_cfg.num_cores,
            report_all_alignments=False,
            seed_length=15,  # -L 22
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
    else:
        raise ValueError(f"Unrecognized aligner `{aligner_type}`")
