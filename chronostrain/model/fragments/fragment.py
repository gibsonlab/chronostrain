"""
  fragment.py
"""
import numpy as np
from typing import List
from chronostrain.util.sequences import Sequence


class Fragment:
    def __init__(self, index: int, seq: Sequence, metadata: List[str] = None):
        self.index = index
        self.seq: Sequence = seq
        self.metadata: List[str] = metadata

    def __eq__(self, other):
        if not isinstance(other, Fragment):
            return False
        else:
            return self.index == other.index and len(self.seq) == len(other.seq) and np.sum(self.seq != other.seq) == 0

    def add_metadata(self, metadata: str):
        if self.metadata is None:
            self.metadata = [metadata]
        else:
            self.metadata.append(metadata)

    def __str__(self):
        acgt_seq = self.seq.nucleotides()
        return "Fragment({}:{})".format(
            '|'.join(self.metadata) if self.metadata else "",
            acgt_seq[:5] + "..." if len(acgt_seq) > 5 else acgt_seq
        )

    def __repr__(self):
        return "Fragment({}:{})".format(
            '|'.join(self.metadata) if self.metadata else "",
            self.seq.nucleotides()
        )

    def __len__(self):
        return len(self.seq)
