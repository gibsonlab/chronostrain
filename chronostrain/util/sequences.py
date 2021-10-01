"""
    sequences.py
    Contains utility functions involving sequences/subsequences.
"""

import numpy as np


# ================ Basic complement/reverse-complement operations, in nucleotide space.

_complement_translation = {
    'A': 'T',
    'a': 'T',
    'T': 'A',
    't': 'A',
    'G': 'C',
    'g': 'C',
    'C': 'G',
    'c': 'G'
}


def _complement_char(nucleotide: str, ignore_keyerror=False):
    """
    Return the complement of the specified nucleotide. If ignore_keyerror is True, return all non-valid
    nucleotides unchanged.
    """
    try:
        return _complement_translation[nucleotide]
    except KeyError:
        if ignore_keyerror:
            return nucleotide
        else:
            raise


def complement_seq(seq: str, ignore_keyerror=False) -> str:
    return ''.join([_complement_char(c, ignore_keyerror=ignore_keyerror) for c in seq])


def reverse_complement_seq(seq: str, ignore_keyerror=False) -> str:
    return ''.join([_complement_char(c, ignore_keyerror=ignore_keyerror) for c in seq[::-1]])


# ================ Cyclic group representation.
_acgt_to_z4 = {
    "A": 0,
    "a": 0,
    "C": 1,
    "c": 1,
    "G": 2,
    "g": 2,
    "T": 3,
    "t": 3,
    "N": 4  # Special character.
}
nucleotide_N_z4 = _acgt_to_z4['N']

_z4_to_acgt = ["A", "C", "G", "T", "N"]

SEQ_DTYPE = np.ubyte


def map_nucleotide_to_z4(nucleotide: str) -> int:
    return _acgt_to_z4[nucleotide]


def map_z4_to_nucleotide(z4: int) -> str:
    return _z4_to_acgt[z4]


def nucleotides_to_z4(nucleotides: str) -> np.ndarray:
    """
    Convert an input nucleotide string (A/C/G/T) to a torch tensor of elements of integers mod 4 (0/1/2/3).
    :param nucleotides:
    :return:
    """
    # Note: This is the fastest known version (despite the for loop), by an order of magnitude.
    # Even faster than np.vectorize(_acgt_to_z4.get)(list(s)), or using numba.jit(nopython=True) with numba.typed.Dict.
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
