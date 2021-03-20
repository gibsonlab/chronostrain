"""
    sequences.py
    Contains utility functions involving sequences/subsequences.
"""

import numpy as np


# Dihedral group representation.
_acgt_to_z4 = {
    "A": 0,
    "a": 0,
    "C": 1,
    "c": 1,
    "G": 2,
    "g": 2,
    "T": 3,
    "t": 3
}

_z4_to_acgt = ["A", "C", "G", "T"]

SEQ_DTYPE = np.ubyte


def nucleotides_to_z4(nucleotides: str) -> np.ndarray:
    """
    Convert an input nucleotide string (A/C/G/T) to a torch tensor of elements of integers mod 4 (0/1/2/3).
    :param nucleotides:
    :return:
    """
    z4seq = np.zeros(shape=len(nucleotides), dtype=SEQ_DTYPE)
    for i, nucleotide in enumerate(nucleotides):
        z4seq[i] = _acgt_to_z4[nucleotide]
    return z4seq


def z4_to_nucleotides(z4seq: np.ndarray) -> str:
    if len(z4seq.shape) > 1:
        raise ValueError("Expected 1-D array, instead got array of size {}.".format(
            z4seq.shape
        ))

    return "".join([
        _z4_to_acgt[element] for element in z4seq
    ])
