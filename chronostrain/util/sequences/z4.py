import numpy as np  # Not using jax

# ================================= Constants.
NucleotideDtype = np.ubyte

# Cyclic group representation.
_acgt_to_z4 = {
    "A": 0,
    "a": 0,
    "C": 1,
    "c": 1,
    "G": 2,
    "g": 2,
    "T": 3,
    "t": 3,
    "N": 4,  # Special character (unknown base)
    "n": 4,
    chr(4): 4,  # Fallback error char for ssw-align
    "-": 5   # Special character (gap)
}
_z4_to_acgt = ["A", "C", "G", "T", "N", "-"]

nucleotide_N_z4 = _acgt_to_z4['N']
nucleotide_GAP_z4 = _acgt_to_z4['-']
_nucleotides = ['A', 'C', 'G', 'T']
_special = ['N', '-']

z4_nucleotides = [_acgt_to_z4[b] for b in _nucleotides]
z4_special = [_acgt_to_z4[b] for b in _special]


# ================================= Function definitions.
def map_nucleotide_to_z4(nucleotide: str) -> int:
    try:
        return _acgt_to_z4[nucleotide]
    except KeyError:
        raise UnknownNucleotideError(nucleotide) from None


def map_z4_to_nucleotide(z4: int) -> str:
    return _z4_to_acgt[z4]


def nucleotides_to_z4(nucleotides: str) -> np.ndarray:
    """
    Convert an input nucleotide string (A/C/G/T) to a torch tensor of elements of integers mod 4 (0/1/2/3).
    :param nucleotides:
    :return:
    """
    return np.fromiter((map_nucleotide_to_z4(x) for x in nucleotides), NucleotideDtype)


def z4_to_nucleotides(z4seq: np.ndarray) -> str:
    if len(z4seq.shape) > 1:
        raise ValueError("Expected 1-D array, instead got array of size {}.".format(
            z4seq.shape
        ))

    return "".join(
        map_z4_to_nucleotide(element) for element in z4seq
    )


def complement_z4(seq: np.ndarray) -> np.ndarray:
    return np.where(
        seq < 4,
        3 - seq,
        seq
    )


def reverse_complement_z4(seq: np.ndarray) -> np.ndarray:
    return complement_z4(seq)[::-1]


class UnknownNucleotideError(Exception):
    def __init__(self, nucleotide: str):
        super().__init__()
        self.nucleotide = nucleotide
