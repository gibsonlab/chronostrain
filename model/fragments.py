"""
  fragments.py
"""
from dataclasses import dataclass


@dataclass
class Fragment:
    seq: str
    index: int


class FragmentSpace:
    """
    A class representing the space of fragments. Serves as a factory for Fragment instances.
    """
    def __init__(self):
        self.ctr = 0
        self.seq_to_frag = dict()

    def contains_seq(self, seq):
        return seq in self.seq_to_frag

    def __create_seq__(self, seq):
        frag = Fragment(seq, self.ctr)
        self.ctr += 1
        return frag

    def add_seq(self, seq: str):
        """
        :param seq: A string to add
        :return:
        """
        if self.contains_seq(seq):
            return self.seq_to_frag[seq]
        else:
            frag = self.__create_seq__(seq)
            self.seq_to_frag[seq] = frag
            return frag

    def get_fragments(self):
        return self.seq_to_frag.values()

    def get_fragment(self, seq):
        try:
            return self.seq_to_frag[seq]
        except KeyError:
            raise KeyError("Sequence '{}' not in dictionary.".format(seq))

    def size(self):
        return self.ctr
