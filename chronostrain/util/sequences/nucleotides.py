"""
    sequences.py
    Contains utility functions involving sequences/subsequences.
"""
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
    return ''.join(_complement_char(c, ignore_keyerror=ignore_keyerror) for c in seq)


def reverse_complement_seq(seq: str, ignore_keyerror=False) -> str:
    return ''.join(_complement_char(c, ignore_keyerror=ignore_keyerror) for c in seq[::-1])



