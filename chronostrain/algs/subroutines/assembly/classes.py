from typing import List

import numpy as np

from chronostrain.model import Marker


class MarkerContig(object):
    """
    A representaton of a particular length-N region of a marker, and the k-ploidy haplotype assembly of that region.
    """
    def __init__(self, marker: Marker, positions: List[int], assembly: np.ndarray):
        """
        :param marker: The marker that this object is representing.
        :param positions: The positions that the assembly matrix's index represents. (Should be sorted in increasing
        order.)
        :param assembly: An (N x k) array of resolved assembly for this contig.
        """
        self.marker = marker

        if len(positions) != assembly.shape[0]:
            raise ValueError("The number of specified positions must match the length of `assembly`.")

        self.positions = positions
        self.assembly = assembly

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
    def get_ploidy(self) -> int:
        return self.assembly.shape[1]
