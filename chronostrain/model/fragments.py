"""
  fragments.py
"""
from dataclasses import dataclass


@dataclass
class Fragment:
    seq: str
    index: int
    metadata: str = None

    def __eq__(self, other):
        return self.index == other.index

    def add_metadata(self, metadata):
        if self.metadata:
            self.metadata = self.metadata + "|" + metadata
        else:
            self.metadata = metadata

    def __str__(self):
        return "Fragment({}:{}:{})".format(
            self.index,
            self.metadata if self.metadata else "",
            self.seq[:5] + "..." if len(self.seq) > 5 else self.seq
        )


class FragmentSpace:
    """
    A class representing the space of fragments. Serves as a factory for Fragment instances.
    """
    def __init__(self):
        self.ctr = 0
        self.seq_to_frag = dict()
        self.frag_list = list()

    def contains_seq(self, seq):
        return seq in self.seq_to_frag

    def __create_seq__(self, seq):
        frag = Fragment(seq, self.ctr)
        self.seq_to_frag[seq] = frag
        self.frag_list.append(frag)
        self.ctr += 1
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
            frag = self.__create_seq__(seq)

        if metadata:
            frag.add_metadata(metadata)
        return frag

    def get_fragments(self):
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
            raise KeyError("Sequence '{}' not in dictionary.".format(seq))

    def get_fragment_by_index(self, idx: int) -> Fragment:
        return self.frag_list[idx]

    def size(self) -> int:
        """
        :return: The number of fragments supported by this space.
        """
        return self.ctr

    def __str__(self):
        return ",".join([str(frag) for frag in self.frag_list])
