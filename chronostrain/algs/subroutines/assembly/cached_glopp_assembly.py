import csv
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np

from chronostrain.model import SequenceRead
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.alignments.sam import SamFlags
from chronostrain.util.cache import ComputationCache
from chronostrain.util.external import run_glopp, sam_to_bam
from chronostrain.util.quality import phred_to_ascii
from chronostrain.util.sequences import *
from ..cache import ReadsComputationCache

from chronostrain.config import create_logger
logger = create_logger(__name__)


_z4_base_ordering: SeqType = nucleotides_to_z4("ACGT-")
_base_to_idx: Dict[NucleotideDtype, int] = {base: idx for idx, base in enumerate(_z4_base_ordering)}


def to_bam(alignment: MarkerMultipleFragmentAlignment, out_path: Path):
    """
    Converts the alignment into a BAM file, but with a minor unconventional change: GAPs are converted into Ns so that
    we can properly call indel variants.
    """
    sam_path = out_path.with_suffix(".sam")
    sam_version = "1.6"
    chronostrain_version = "empty"  # TODO insert explicit versioning.

    def write_read(read: SequenceRead,
                   map_first_idx: int,
                   map_last_idx: int,
                   revcomp: bool,
                   sam_flags: List[SamFlags], w: csv.writer):
        # Flag calculation (Bitwise OR)
        read_flag = 0
        for flag in sam_flags:
            read_flag = read_flag | flag.value

        # Mapping positions
        aln = alignment.get_alignment(read, revcomp, delete_double_gaps=False)
        query_map_len = map_last_idx - map_first_idx + 1
        assert query_map_len == len(read)

        # Mapping quality
        mapq: int = 255  # (not available)
        rnext: str = "*"
        pnext: int = 0
        tlen: int = 0

        # Query seqs (Convert gaps into N's.)
        query_seq = aln[1, map_first_idx:map_last_idx+1].copy()
        quality = np.zeros(shape=query_seq.shape, dtype=float)
        quality[query_seq == nucleotide_GAP_z4] = 0
        quality[query_seq != nucleotide_GAP_z4] = read.quality
        query_seq[query_seq == nucleotide_GAP_z4] = nucleotide_N_z4

        w.writerow([
            read.id,
            read_flag,
            alignment.marker.id,
            str(map_first_idx + 1),  # 1-indexed mapping position
            str(mapq),
            f"{query_map_len}M",
            rnext,
            str(pnext),
            str(tlen),
            z4_to_nucleotides(query_seq),
            phred_to_ascii(quality, "fastq")
        ])

    with open(sam_path, "w") as f:
        tsv = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        # ========== METADATA.
        tsv.writerow(["@HD", f"VN:{sam_version}", "SO:unsorted"])
        tsv.writerow(["@SQ", f"SN:{alignment.marker.id}", f"LN:{len(alignment.marker.seq)}"])
        tsv.writerow(["@PG", "ID:chronostr", f"VN:{chronostrain_version}", "PN:chronostrain", "CL:empty"])
        tsv.writerow(["@CO", f"Chronostrain BAM from multiple alignment (marker: {alignment.marker.id})"])

        # ========== SEQUENCE ALIGNMENTS.
        entries = [
            (read_obj, False, *alignment.aln_gapped_boundary(read_obj, False))
            for read_obj, r_idx in alignment.forward_read_index_map.items()
        ] + [
            (read_obj, True, *alignment.aln_gapped_boundary(read_obj, True))
            for read_obj, r_idx in alignment.reverse_read_index_map.items()
        ]  # Tuple of (Read, Rev_comp, Start_idx, End_idx).

        entries.sort(key=lambda x: x[2])

        for read_obj, revcomp, start_idx, end_idx in entries:
            flags = [SamFlags.SeqReverseComplement] if revcomp else []
            write_read(read_obj, start_idx, end_idx, revcomp, flags, tsv)

    # now create the BAM file via compression.
    logger.debug(f"Compression SAM ({str(sam_path.name)}) to BAM ({str(out_path.name)}).")
    sam_to_bam(sam_path, out_path)


def to_vcf(alignment: MarkerMultipleFragmentAlignment, variant_counts: np.ndarray, ploidy: int, out_path: Path):
    """
    :param alignment: The multiple alignment instance to use.
    :param variant_counts: The (N x 5) matrix of variants, where each row stores the number of occurrences of each base,
    where the columns are indexed as [A, C, G, T, -].
    :param ploidy: the desired ploidy to run glopp with.
    :param out_path: The path to save the VCF to.
    """

    def render_base(base: NucleotideDtype) -> str:
        if base == nucleotide_GAP_z4:
            return "*"
        return map_z4_to_nucleotide(base)

    with open(out_path, "w") as f:
        tsv = csv.writer(f, quotechar='', delimiter='\t', quoting=csv.QUOTE_NONE)
        # Header rows
        tsv.writerow(["##fileformat=VCFv4.2"])
        tsv.writerow([f"##contig=<ID={alignment.marker.id}>"])
        # tsv.writerow(["##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of samples with Data\">"])
        tsv.writerow(["##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Total number of variants identified\">"])
        tsv.writerow(["##INFO=<ID=AC,Number=A,Type=Integer,Description=\"Read count for each ALT variant\">"])
        tsv.writerow(["##INFO=<ID=INS,Number=0,Type=Flag,Description=\"Insertion into marker\">"])
        tsv.writerow(["##INFO=<ID=DEL,Number=0,Type=Flag,Description=\"At least one of ALT is deletion from marker\">"])
        tsv.writerow(["##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"])
        tsv.writerow([
            "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "VARIANT_GENOTYPE"
        ])

        idx_gap = _base_to_idx[nucleotide_GAP_z4]

        for idx in range(alignment.num_bases()):
            ref_base_z4 = alignment.aligned_marker_seq[idx]
            ref_base_idx = _base_to_idx[ref_base_z4]
            variant_counts_i = variant_counts[idx]

            # Compute the supported variants, not equal to the reference base.
            supported_variant_indices = np.where(variant_counts_i > 0)[0]
            supported_variant_indices = [i for i in supported_variant_indices if i != ref_base_idx]

            # No reads map to this position. Nothing to do.
            if (
                    len(supported_variant_indices) == 0  # no supported variants other than ref.
                    or (len(supported_variant_indices) == 1 and supported_variant_indices[0] == idx_gap)  # only gaps.
            ):
                continue

            # Info tags.
            info_tags = [
                f"AC={','.join(str(variant_counts_i[v]) for v in supported_variant_indices)}",
                f"AN={len(supported_variant_indices)}"
            ]
            if ref_base_z4 == nucleotide_GAP_z4:
                info_tags.append("INS")
            elif variant_counts_i[-1] > 0:  # Assumes that "-" is indexed at the last column!
                info_tags.append("DEL")

            # Write the row to file.
            tsv.writerow([
                alignment.marker.id,  # chrom
                idx + 1,  # pos
                ".",  # id
                map_z4_to_nucleotide(ref_base_z4),  # ref
                ",".join(render_base(_z4_base_ordering[v]) for v in supported_variant_indices),  # alt
                "100",  # qual
                "PASS",  # filter
                ";".join(info_tags),  # info
                "GT",  # FORMAT
                "/".join("." for _ in range(ploidy))
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

        self.relative_dir = Path(f"glopp/{self.alignment.marker.id}")
        self.absolute_dir = self.cache.cache_dir / self.relative_dir
        self.variant_counts = self.count_variants()

    def count_variants(self) -> np.ndarray:
        """
        :return: An (N x 5) array of variant counts.
        """
        counts = np.zeros(shape=(self.alignment.num_bases(), len(_z4_base_ordering)), dtype=int)
        for r in range(self.alignment.read_multi_alignment.shape[0]):
            row = self.alignment.read_multi_alignment[r]
            first_idx, last_idx = self.alignment.aln_gapped_boundary_of_row(r)
            row = row[first_idx:last_idx + 1]
            for b_idx, z4base in enumerate(_z4_base_ordering):
                z4_base_locs = np.where(row == z4base)[0]
                counts[first_idx + z4_base_locs, b_idx] += 1
        return counts

    def prepare_glopp_input(self, ploidy: int) -> Tuple[Path, Path]:
        bam_path = self.prepare_bam(ploidy)
        vcf_path = self.prepare_vcf(self.variant_counts, ploidy)
        return bam_path, vcf_path

    def run_glopp(self, num_strains: int):
        subdir = f"ploidy_{num_strains}/output"
        phasing_rel_output_dir = self.relative_dir / subdir
        phasing_abs_output_dir = self.absolute_dir / subdir
        expected_rel_output_path = phasing_rel_output_dir / f"{self.alignment.marker.id}_phasing.txt"

        def _call():
            bam, vcf = self.prepare_glopp_input(ploidy=num_strains)
            run_glopp(
                bam_path=bam,
                vcf_path=vcf,
                output_dir=phasing_abs_output_dir,
                ploidy=num_strains
            )
            return "todo return variants here"

        self.cache.call(
            relative_filepath=expected_rel_output_path,
            fn=_call,
            save=lambda p, o: None,
            load=lambda p: "todo parse variants from glopp output here"
        )

    def prepare_bam(self, ploidy: int) -> Path:
        target_path = self.absolute_dir / f"ploidy_{ploidy}" / "alignments.bam"
        target_path.parent.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Creating BAM file {str(target_path)}.")
        to_bam(self.alignment, target_path)
        return target_path

    def prepare_vcf(self, variants: np.array, ploidy: int) -> Path:
        target_path = self.absolute_dir / f"ploidy_{ploidy}" / "variants.vcf"
        target_path.parent.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Creating VCF file {str(target_path)}.")
        to_vcf(self.alignment, variants, ploidy, target_path)
        return target_path
