from typing import List, Tuple, Union

import numpy as np

from chronostrain.model import Marker
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.sequences import SeqType, nucleotide_GAP_z4

from chronostrain.config import create_logger
logger = create_logger(__name__)


def remove_gaps(seq: SeqType) -> SeqType:
    return seq[seq != nucleotide_GAP_z4]


class FloppMarkerAssembly(object):
    def __init__(self, aln: MarkerMultipleFragmentAlignment, contigs: List['FloppMarkerContig'], all_positions: np.ndarray):
        self.aln = aln
        self.contigs = contigs
        self.all_positions = all_positions

        # Map relative positioning in concatenated array.
        strand_counts = np.array([contig.num_strands for contig in contigs])
        self.num_total_strands = np.sum(strand_counts)
        self.cumulative_strand_counts = np.cumsum(strand_counts)

    @property
    def marker(self) -> Marker:
        return self.aln.marker

    @property
    def num_contigs(self) -> int:
        return len(self.contigs)

    def mean_counts(self) -> np.ndarray:
        return np.concatenate([
            contig.mean_counts
            for contig in self.contigs
        ], axis=0)

    def get_contig_of(self, concatenated_strand_index: int) -> Tuple['FloppMarkerContig', int]:
        contig_idx = np.where(concatenated_strand_index < self.cumulative_strand_counts)[0][0]
        if contig_idx == 0:
            relative_idx = concatenated_strand_index
        else:
            relative_idx = concatenated_strand_index - self.cumulative_strand_counts[contig_idx - 1]
        return self.contigs[contig_idx], relative_idx

    def contig_base_seq(self, contigs_strands: List[Union[int, None]]) -> Tuple[SeqType, int]:
        seq = self.aln.aligned_marker_seq.copy()
        read_count = 0
        for contig_idx, strand_idx in enumerate(contigs_strands):
            contig: FloppMarkerContig = self.contigs[contig_idx]
            # Only copy from strand variants if strand_idx is specified. Otherwise, leave the base sequence.
            if strand_idx is not None:
                seq[contig.positions] = contig.get_strand(strand_idx)
                read_count += contig.num_reads_of_strand(strand_idx)
        return remove_gaps(seq), read_count


class FloppMarkerContig(object):
    """
    A representaton of a particular length-N region of a marker, and the k-ploidy haplotype assembly of that region.
    """
    def __init__(self,
                 marker: Marker,
                 contig_idx: int,
                 positions: np.ndarray,
                 assembly: np.ndarray,
                 counts: np.ndarray,
                 num_reads_per_strand: np.ndarray):
        """
        :param marker: The marker that this object is representing.
        :param positions: The list of integer-valued positions that the assembly matrix's index represents.
            (Should be sorted in increasing order.)
        :param assembly: An (N x k) array of resolved assembly for this contig.
        :param counts: An (N x k x T) array of marginal counts the number of reads mapped to the k-th "haplotype"
            at position N at time t.
        """
        self.marker = marker
        self.positions = positions
        self.contig_idx = contig_idx

        # The primary contents of this object.
        if len(positions) != assembly.shape[0]:
            raise ValueError("The number of specified positions must match the length of `assembly`.")

        if assembly.shape[0] != counts.shape[0] or assembly.shape[1] != counts.shape[1]:
            raise ValueError("The shape of `assembly` must match the first two dims of `counts`.")
        self.assembly = assembly
        self.counts = counts
        self.mean_counts = np.mean(self.counts, axis=0)  # (k x T).
        self.num_reads_per_strand = num_reads_per_strand

        # Trim zero count strands.
        num_zero_count_strands = np.sum(self.num_reads_per_strand == 0)
        if num_zero_count_strands > 0:
            logger.debug(
                f"{marker.id}, Contig {contig_idx} - Trimming {num_zero_count_strands} strands with zero count."
            )

            support_indices = np.where(self.num_reads_per_strand > 0)[0]
            self.mean_counts = self.mean_counts[support_indices, :]
            self.assembly = self.assembly[:, support_indices]
            self.counts = self.counts[:, support_indices, :]
            self.num_reads_per_strand = self.num_reads_per_strand[support_indices]

    @property
    def leftmost_pos(self) -> int:
        return self.positions[0]

    @property
    def rightmost_pos(self) -> int:
        return self.positions[-1]

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def num_strands(self) -> int:
        return self.assembly.shape[1]

    def get_strand(self, idx: int) -> SeqType:
        return self.assembly[:, idx]

    def num_reads_of_strand(self, strand_idx: int) -> int:
        return self.num_reads_per_strand[strand_idx]
