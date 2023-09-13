from abc import abstractmethod
import numpy as np
from .z4 import reverse_complement_z4, z4_to_nucleotides, nucleotide_N_z4


class Sequence(object):
    @abstractmethod
    def nucleotides(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def bytes(self) -> np.ndarray:
        raise NotImplementedError()

    def revcomp_bytes(self) -> np.ndarray:
        return reverse_complement_z4(self.bytes())

    def revcomp_nucleotides(self) -> str:
        return z4_to_nucleotides(self.revcomp_bytes())

    def number_of_ns(self) -> int:
        return np.sum(np.equal(self.bytes(), nucleotide_N_z4))

    @abstractmethod
    def revcomp_seq(self) -> 'Sequence':
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __str__(self):
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError()
