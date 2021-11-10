import numpy as np

# ================================= Constants.
NucleotideDtype = np.ubyte
SeqType = np.ndarray

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
    return _acgt_to_z4[nucleotide]


def map_z4_to_nucleotide(z4: int) -> str:
    return _z4_to_acgt[z4]


def nucleotides_to_z4(nucleotides: str) -> SeqType:
    """
    Convert an input nucleotide string (A/C/G/T) to a torch tensor of elements of integers mod 4 (0/1/2/3).
    :param nucleotides:
    :return:
    """
    # Note: This is the fastest known version (despite the for loop), by an order of magnitude.
    # Even faster than np.vectorize(_acgt_to_z4.get)(list(s)), or using numba.jit(nopython=True) with numba.typed.Dict.
    z4seq = np.zeros(shape=len(nucleotides), dtype=NucleotideDtype)
    for i, nucleotide in enumerate(nucleotides):
        z4seq[i] = _acgt_to_z4[nucleotide]
    return z4seq


def z4_to_nucleotides(z4seq: SeqType) -> str:
    if len(z4seq.shape) > 1:
        raise ValueError("Expected 1-D array, instead got array of size {}.".format(
            z4seq.shape
        ))

    return "".join(
        _z4_to_acgt[element] for element in z4seq
    )
