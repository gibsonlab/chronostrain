"""
  fragments.py
"""
import numpy as np
from typing import Dict, List, Iterable

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.util.sequences import z4_to_nucleotides, SeqType


class Fragment:
    def __init__(self, seq: SeqType, index: int, metadata: List[str] = None):
        self.seq: SeqType = seq
        self.seq_len = len(seq)
        self.index: int = index
        self.metadata: List[str] = metadata

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        if not isinstance(other, Fragment):
            return False
        else:
            return self.index == other.index \
                   and len(self.seq) == len(other.seq) \
                   and np.sum(self.seq != other.seq) == 0

    def add_metadata(self, metadata: str):
        if self.metadata is None:
            self.metadata = [metadata]
        else:
            self.metadata.append(metadata)

    def nucleotide_content(self) -> str:
        return z4_to_nucleotides(self.seq)

    def to_seqrecord(self, description: str = "") -> SeqRecord:
        return SeqRecord(
            Seq(self.nucleotide_content()),
            id="FRAGMENT_{}".format(self.index),
            description=description
        )

    def __str__(self):
        acgt_seq = self.nucleotide_content()
        return "Fragment({}:{}:{})".format(
            self.index,
            '|'.join(self.metadata) if self.metadata else "",
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
        self.min_frag_len = 0

    def contains_seq(self, seq: SeqType) -> bool:
        return self.seq_to_key(seq) in self.seq_to_frag

    @staticmethod
    def seq_to_key(seq: SeqType) -> str:
        return str(seq)

    def _create_frag(self, seq: SeqType):
        frag = Fragment(seq=seq, index=self.fragment_instances_counter)
        self.seq_to_frag[self.seq_to_key(seq)] = frag
        self.frag_list.append(frag)
        self.fragment_instances_counter += 1

        if len(self) == 0 or self.min_frag_len > len(seq):
            self.min_frag_len = len(seq)

        return frag

    def add_seq(self, seq: SeqType, metadata: str = None) -> Fragment:
        """
        Tries to add a new Fragment instance encapsulating the string seq.
        If the seq is already in the space, nothing happens.

        :param seq: A string to add to the space.
        :param metadata: The fragment-specific metadata to add (useful for record-keeping on toy data).
        :return: the Fragment instance that got added.
        """
        if self.contains_seq(seq):
            frag = self.get_fragment(seq)
        else:
            frag = self._create_frag(seq)

        if metadata is not None:
            frag.add_metadata(metadata)
        return frag

    def get_fragments(self) -> Iterable[Fragment]:
        return self.seq_to_frag.values()

    def get_fragment(self, seq: SeqType) -> Fragment:
        """
        Retrieves a Fragment instance corresponding to the sequence.
        Raises KeyError if the sequence is not found in the space.

        :param seq: A string.
        :return: the Fragment instance encapsulating the seq.
        """
        try:
            return self.seq_to_frag[self.seq_to_key(seq)]
        except KeyError:
            raise KeyError("Sequence query (len={}) not in dictionary.".format(len(seq))) from None

    def get_fragment_by_index(self, idx: int) -> Fragment:
        return self.frag_list[idx]

    def size(self) -> int:
        """
        :return: The number of fragments supported by this space.
        """
        return len(self.frag_list)

    def merge_with(self, other: 'FragmentSpace'):
        num_new_frags = 0
        for other_frag in other.frag_list:
            if self.contains_seq(other_frag.seq):
                continue

            new_frag = Fragment(
                seq=other_frag.seq,
                index=self.fragment_instances_counter + other_frag.index,
                metadata=other_frag.metadata
            )
            self.frag_list.append(new_frag)
            self.seq_to_frag[self.seq_to_key(other_frag.seq)] = new_frag
            num_new_frags += 1
        self.fragment_instances_counter += num_new_frags

    def __str__(self):
        return ",".join(str(frag) for frag in self.frag_list)

    def __repr__(self):
        return ",".join(repr(frag) for frag in self.frag_list)

    def __iter__(self):
        return self.frag_list.__iter__()

    def __len__(self):
        return self.size()
