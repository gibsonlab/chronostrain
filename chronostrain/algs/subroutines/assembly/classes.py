import numpy as np

from chronostrain.model import Marker
from chronostrain.util.sequences import SeqType


class MarkerContig(object):
    """
    A representaton of a particular length-N region of a marker, and the k-ploidy haplotype assembly of that region.
    """
    def __init__(self,
                 marker: Marker,
                 contig_idx: int,
                 positions: np.ndarray,
                 assembly: np.ndarray,
                 counts: np.ndarray):
        """
        :param marker: The marker that this object is representing.
        :param positions: The list of integer-valued positions that the assembly matrix's index represents.
            (Should be sorted in increasing order.)
        :param assembly: An (N x k) array of resolved assembly for this contig.
        :param counts: An (N x k x T) array of marginal counts the number of reads mapped to the k-th "haplotype"
            at position N at time t.
        """
        self.marker = marker

        if len(positions) != assembly.shape[0]:
            raise ValueError("The number of specified positions must match the length of `assembly`.")

        if assembly.shape[0] != counts.shape[0] or assembly.shape[1] != counts.shape[1]:
            raise ValueError("The shape of `assembly` must match the first two dims of `counts`.")

        self.positions = positions
        self.contig_idx = contig_idx
        self.assembly = assembly
        self.counts = counts

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
    def ploidy(self) -> int:
        return self.assembly.shape[1]

    def get_strand(self, idx: int) -> SeqType:
        return self.assembly[:, idx]

    def mean_counts(self) -> np.ndarray:
        return np.mean(self.counts, axis=0)
