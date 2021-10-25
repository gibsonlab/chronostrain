import csv
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from chronostrain.model import SequenceRead
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.alignments.sam import SamFlags
from chronostrain.util.cache import ComputationCache
from chronostrain.util.external import run_glopp
from chronostrain.util.quality import phred_to_ascii
from chronostrain.util.sequences import nucleotide_GAP_z4, nucleotide_N_z4, z4_to_nucleotides, map_z4_to_nucleotide
from ..cache import ReadsComputationCache

from chronostrain.config import create_logger
logger = create_logger(__name__)


def to_bam(alignment: MarkerMultipleFragmentAlignment, out_path: Path):
    """
    Converts the alignment into a BAM file, but with a minor unconventional change: GAPs are converted into Ns so that
    we can properly call indel variants.
    """
    bam_version = "1.6"
    chronostrain_version = "empty"  # TODO insert explicit versioning.

    def write_read(read: SequenceRead, revcomp: bool, sam_flags: List[SamFlags], w: csv.writer):
        # Flag calculation (Bitwise OR)
        read_flag = 0
        for flag in sam_flags:
            read_flag = read_flag | flag.value

        # Mapping positions
        aln = alignment.get_alignment(read, revcomp, delete_double_gaps=False)
        query_ungapped_positions = np.where(aln[1] != nucleotide_GAP_z4)[0]
        ref_map_start_idx: int = query_ungapped_positions[0]
        ref_map_end_idx = query_ungapped_positions[-1]
        query_map_len = ref_map_end_idx - ref_map_start_idx + 1
        assert query_map_len == len(read)

        # Mapping quality
        mapq: int = 255  # (not available)
        rnext: str = "*"
        pnext: int = 0
        tlen: int = 0

        # Query seqs (Convert gaps into N's.)
        query_seq = aln[1, ref_map_start_idx:ref_map_end_idx+1].copy()
        quality = np.zeros(shape=query_seq.shape, dtype=float)
        quality[query_seq == nucleotide_GAP_z4] = 0
        quality[query_seq != nucleotide_GAP_z4] = read.quality
        query_seq[query_seq == nucleotide_GAP_z4] = nucleotide_N_z4

        w.writerow([
            read.id,
            read_flag,
            alignment.marker.id,
            str(ref_map_start_idx + 1),
            str(mapq),
            f"{query_map_len}M",
            rnext,
            str(pnext),
            str(tlen),
            z4_to_nucleotides(query_seq),
            phred_to_ascii(quality, "fastq")
        ])

    with open(out_path, "w") as f:
        tsv = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        # ========== METADATA.
        tsv.writerow(["@HD", f"VN:{bam_version}", "SO:unsorted"])
        tsv.writerow(["@SQ", f"SN:{alignment.marker.id}", f"LN:{len(alignment.marker.seq)}"])
        tsv.writerow(["@PG", "ID:chronostr", f"VN:{chronostrain_version}", "PN:chronostrain", "CL:empty"])
        tsv.writerow(["@CO", "Chronostrain BAM from multiple alignment (marker: {self.marker.id})"])

        # ========== SEQUENCE ALIGNMENTS.
        for read_obj, r_idx in alignment.forward_read_index_map.items():
            flags = []
            write_read(read_obj, False, flags, tsv)
        for read_obj, r_idx in alignment.reverse_read_index_map.items():
            flags = [SamFlags.SeqReverseComplement]
            write_read(read_obj, True, flags, tsv)


def to_vcf(alignment: MarkerMultipleFragmentAlignment, variant_counts: np.ndarray, out_path: Path):
    """
    :param alignment: The multiple alignment instance to use.
    :param variant_counts: The (N x 5) matrix of variants, where each row stores the number of occurrences of each base,
    where the columns are indexed as [A, C, G, T, -].
    :param out_path: The path to save the VCF to.
    """
    idx_to_base = ["A", "C", "G", "T", "-"]
    base_to_idx = {base: idx for idx, base in enumerate(idx_to_base)}

    def render_base(base: str) -> str:
        if base == "-":
            return "*"
        return base

    with open(out_path, "w") as f:
        tsv = csv.writer(f, quotechar='', delimiter='\t', quoting=csv.QUOTE_NONE)
        # Header rows
        tsv.writerow(["##fileformat=VCFv4.2"])
        tsv.writerow(["##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of samples with Data\">"])
        tsv.writerow(["##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Total number of variants identified\">"])
        tsv.writerow(["##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Allele count for each ALT variant\">"])
        tsv.writerow(["##INFO=<ID=INS,Number=0,Type=Flag,Description=\"Insertion into marker\">"])
        tsv.writerow(["##INFO=<ID=DEL,Number=0,Type=Flag,Description=\"At least one of ALT is deletion from marker\">"])
        tsv.writerow([
            "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"
        ])

        for idx in range(alignment.num_bases()):
            ref_base = alignment.aligned_marker_seq[idx]
            ref_base_idx = base_to_idx[ref_base]
            variant_counts_i = variant_counts[idx]

            # Compute the supported variants, not equal to the reference base.
            supported_variant_indices = np.where(variant_counts_i > 0)[0]
            supported_variant_indices = [i for i in supported_variant_indices if i != ref_base_idx]

            # No reads map to this position. Nothing to do.
            if len(supported_variant_indices) == 1 and supported_variant_indices[0] == nucleotide_GAP_z4:
                continue

            # Info tags.
            info_tags = [
                f"AC={','.join(variant_counts[v] for v in supported_variant_indices)}",
                f"AN={len(supported_variant_indices)}"
            ]
            if ref_base == nucleotide_GAP_z4:
                info_tags.append("INS")
            elif variant_counts_i[-1] > 0: # Assumes that "-" is indexed at the last column!
                info_tags.append("DEL")

            # Write the row to file.
            tsv.writerow([
                alignment.marker.id,  # chrom
                idx + 1,  # pos
                ".",  # id
                map_z4_to_nucleotide(ref_base),  # ref
                ",".join(render_base(idx_to_base[v]) for v in supported_variant_indices),  # alt
                "100",  # qual
                "PASS",  # filter
                ";".join(info_tags)  # info
            ])


class CachedGloppVariantAssembly(object):
    def __init__(self,
                 reads: TimeSeriesReads,
                 alignment: MarkerMultipleFragmentAlignment,
                 cache_override: Optional[ComputationCache] = None
                 ):
        self.reads = reads

        if cache_override is not None:
            self.cache = cache_override
        else:
            self.cache = ReadsComputationCache(reads)

        self.alignment: MarkerMultipleFragmentAlignment = alignment

        self.relative_input_dir = Path(f"glopp/input/{self.alignment.marker.id}")
        self.relative_output_dir = Path(f"glopp/output/{self.alignment.marker.id}")
        self.bam_path = self.relative_input_dir / "alignments.bam"
        self.vcf_path = self.relative_input_dir / "variants.vcf"

        self.glopp_output_dir = self.cache.cache_dir / "glopp" / "output"
        self.variant_counts = self.count_variants()

    def count_variants(self) -> np.ndarray:
        pass

    def prepare_glopp_input(self) -> Tuple[Path, Path]:
        bam_path = self.prepare_bam()
        vcf_path = self.prepare_vcf(self.variant_counts)
        return bam_path, vcf_path

    def prepare_bam(self) -> Path:
        out_path = self.glopp_output_dir / "alignments.bam"
        to_bam(self.alignment, out_path)
        return out_path

    def prepare_vcf(self, variants: np.array) -> Path:
        out_path = self.glopp_output_dir / "variants."
        to_vcf(self.alignment, variants, out_path)
        return out_path
