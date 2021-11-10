from typing import List, Tuple, Union

import numpy as np

from chronostrain.model import Marker
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.sequences import SeqType

from chronostrain.config import create_logger
logger = create_logger(__name__)


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
        return seq, read_count


class FloppMarkerContig(object):
    """
    A representaton of a particular length-N region of a marker, and the k-ploidy haplotype assembly of that region.
    """
    def __init__(self,
                 marker: Marker,
                 contig_idx: int,
                 positions: np.ndarray,
                 assembly: np.ndarray,
                 read_counts: np.ndarray):
        """
        :param marker: The marker that this object is representing.
        :param positions: The list of integer-valued positions that the assembly matrix's index represents.
            (Should be sorted in increasing order.)
        :param assembly: An (N x k) array of resolved assembly for this contig.
        :param read_counts: An (k x T) array of read counts per each strand, per timepoint.
        """
        self.marker = marker
        self.positions = positions
        self.contig_idx = contig_idx

        # The primary contents of this object.
        if len(positions) != assembly.shape[0]:
            raise ValueError("The number of specified positions must match the length of `assembly`.")

        self.assembly = assembly
        self.read_counts = read_counts

        # Trim zero count strands.
        num_reads_per_strand = np.sum(read_counts, axis=1)
        num_zero_count_strands = np.sum(num_reads_per_strand == 0)
        if num_zero_count_strands > 0:
            logger.debug(
                f"{marker.id}, Contig {contig_idx} - Trimming {num_zero_count_strands} strands with zero count."
            )

            support_indices = np.where(num_reads_per_strand > 0)[0]
            self.assembly = self.assembly[:, support_indices]
            self.read_counts = self.read_counts[support_indices, :]

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
        return self.read_counts[strand_idx, :].sum()
