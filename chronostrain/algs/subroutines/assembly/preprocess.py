import csv
from pathlib import Path
from typing import List, Dict

import numpy as np

from chronostrain.model import SequenceRead
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.alignments.sam import SamFlags
from chronostrain.util.external import sam_to_bam
from chronostrain.util.quality import phred_to_ascii
from chronostrain.util.sequences import *

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
                   reverse_complement: bool,
                   sam_flags: List[SamFlags], w: csv.writer):
        # Flag calculation (Bitwise OR)
        read_flag = 0
        for flag in sam_flags:
            read_flag = read_flag | flag.value

        # Mapping positions
        aln = alignment.get_alignment(read, reverse_complement, delete_double_gaps=False)
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