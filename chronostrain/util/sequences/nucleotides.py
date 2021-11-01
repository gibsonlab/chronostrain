"""
    sequences.py
    Contains utility functions involving sequences/subsequences.
"""
from typing import Dict
import numpy as np

from .z4 import map_nucleotide_to_z4, SeqType, NucleotideDtype

# ================ Basic complement/reverse-complement operations, in nucleotide space.
_complement_translation: Dict[int, int] = {
    map_nucleotide_to_z4('A'): map_nucleotide_to_z4('T'),
    map_nucleotide_to_z4('T'): map_nucleotide_to_z4('A'),
    map_nucleotide_to_z4('G'): map_nucleotide_to_z4('C'),
    map_nucleotide_to_z4('C'): map_nucleotide_to_z4('G')
}


def _complement_char(nucleotide: int) -> int:
    """
    Return the complement of the specified nucleotide. If ignore_keyerror is True, return all non-valid
    nucleotides unchanged.
    """
    return _complement_translation[nucleotide]


def complement_seq(seq: SeqType) -> SeqType:
    return np.array([
        _complement_char(x) for x in seq
    ], dtype=NucleotideDtype)


def reverse_complement_seq(seq: SeqType) -> str:
    return complement_seq(seq)[::-1]



