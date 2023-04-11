from typing import Union

import numpy as np
from .base import Sequence
from .z4 import nucleotides_to_z4, z4_to_nucleotides, reverse_complement_z4


class AllocatedSequence(Sequence):
    """
    Represents a single sequence, whose contents (nucleotides) are stored as a byte array in memory.
    """
    def __init__(self, seq: Union[str, np.ndarray]):
        if isinstance(seq, str):
            self.byte_seq = nucleotides_to_z4(seq)
        elif isinstance(seq, np.ndarray):
            self.byte_seq = seq
        else:
            raise ValueError("Unrecognized seq argument type: {}".format(type(seq)))

    def nucleotides(self) -> str:
        return z4_to_nucleotides(self.byte_seq)

    def bytes(self) -> np.ndarray:
        return self.byte_seq

    def revcomp_seq(self) -> 'AllocatedSequence':
        return AllocatedSequence(reverse_complement_z4(self.byte_seq))

    def __len__(self):
        return len(self.byte_seq)

    def __str__(self):
        return self.nucleotides()

    def __repr__(self):
        return self.nucleotides()
