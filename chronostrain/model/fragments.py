"""
  fragments.py
"""
from typing import Dict, List, Iterable

import numpy as np
from chronostrain.util.sequences import nucleotides_to_z4, z4_to_nucleotides


class Fragment:
    def __init__(self, seq: str, index: int, metadata: str):
        self.seq: np.ndarray = nucleotides_to_z4(seq)
        self.seq_len = len(seq)
        self.index: int = index
        self.metadata: str = metadata

    def __hash__(self):
        return hash(self.seq)

    def __eq__(self, other):
        if not isinstance(other, Fragment):
            return False
        else:
            return self.index == other.index

    def add_metadata(self, metadata):
        if self.metadata:
            self.metadata = self.metadata + "|" + metadata
        else:
            self.metadata = metadata

    def nucleotide_content(self) -> str:
        return z4_to_nucleotides(self.seq)

    def __str__(self):
        acgt_seq = self.nucleotide_content()
        return "Fragment({}:{}:{})".format(
            self.index,
            self.metadata if self.metadata else "",
            acgt_seq[:5] + "..." if len(acgt_seq) > 5 else acgt_seq
        )

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.seq_len


class FragmentSpace:
    """
    A class representing the space of fragments. Serves as a factory for Fragment instances.
    """
    def __init__(self):
        self.fragment_instances_counter = 0
        self.seq_to_frag: Dict[str, Fragment] = dict()
        self.frag_list: List[Fragment] = list()

    def contains_seq(self, seq):
        return seq in self.seq_to_frag

    def _create_seq(self, seq):
        frag = Fragment(seq=seq, index=self.fragment_instances_counter, metadata="")
        self.seq_to_frag[seq] = frag
        self.frag_list.append(frag)
        self.fragment_instances_counter += 1
        return frag

    def add_seq(self, seq: str, metadata: str = None):
        """
        Tries to add a new Fragment instance encapsulating the string seq.
        If the seq is already in the space, nothing happens.

        :param seq: A string to add to the space.
        :param metadata: The fragment-specific metadata to add (useful for record-keeping on toy data).
        :return: the Fragment instance that got added.
        """
        if self.contains_seq(seq):
            frag = self.seq_to_frag[seq]
        else:
            frag = self._create_seq(seq)

        if metadata:
            frag.add_metadata(metadata)
        return frag

    def get_fragments(self) -> Iterable[Fragment]:
        return self.seq_to_frag.values()

    def get_fragment(self, seq: str) -> Fragment:
        """
        Retrieves a Fragment instance corresponding to the sequence.
        Raises KeyError if the sequence is not found in the space.

        :param seq: A string.
        :return: the Fragment instance encapsulating the seq.
        """
        try:
            return self.seq_to_frag[seq]
        except KeyError:
            raise KeyError("Sequence '{}' not in dictionary.".format(seq)) from None

    def get_fragment_by_index(self, idx: int) -> Fragment:
        return self.frag_list[idx]

    def size(self) -> int:
        """
        :return: The number of fragments supported by this space.
        """
        return len(self.frag_list)

    def __str__(self):
        return ",".join([str(frag) for frag in self.frag_list])

    def __iter__(self):
        return self.frag_list.__iter__()

    def __len__(self):
        return self.size()
