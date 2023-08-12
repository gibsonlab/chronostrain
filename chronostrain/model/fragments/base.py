"""
  base.py
"""
from pathlib import Path

import numpy as np
from typing import Dict, List, Iterable, Iterator, Tuple

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.util.sequences import Sequence


class Fragment:
    def __init__(self, seq: Sequence, index: int, metadata: List[str] = None):
        self.seq: Sequence = seq
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

    def __str__(self):
        acgt_seq = self.seq.nucleotides()
        return "Fragment({}:{}:{})".format(
            self.index,
            '|'.join(self.metadata) if self.metadata else "",
            acgt_seq[:5] + "..." if len(acgt_seq) > 5 else acgt_seq
        )

    def __repr__(self):
        return "Fragment({}:{}:{})".format(
            self.index,
            '|'.join(self.metadata) if self.metadata else "",
            self.seq.nucleotides()
        )

    def __len__(self):
        return len(self.seq)


class FragmentSpace:
    """
    A class representing the space of fragments. Serves as a factory for Fragment instances.
    """
    def __init__(self):
        self.fragment_instances_counter = 0
        self.seq_to_frag: Dict[str, Fragment] = dict()
        self.frag_list: List[Fragment] = list()
        self.min_frag_len = 0

    def _contains_seq(self, seq: Sequence) -> bool:
        return self._seq_to_key(seq) in self.seq_to_frag

    @staticmethod
    def _seq_to_key(seq: Sequence) -> str:
        return seq.nucleotides()

    def _create_frag(self, seq: Sequence):
        frag = Fragment(seq=seq, index=self.fragment_instances_counter)
        self.seq_to_frag[self._seq_to_key(seq)] = frag
        self.frag_list.append(frag)
        self.fragment_instances_counter += 1

        if len(self) == 0 or self.min_frag_len > len(seq):
            self.min_frag_len = len(seq)

        return frag

    def add_seq(self, seq: Sequence, metadata: str = None) -> Fragment:
        """
        Tries to add a new Fragment instance encapsulating the string seq.
        If the seq is already in the space, nothing happens.

        :param seq: A string to add to the space.
        :param metadata: The fragment-specific metadata to add (useful for record-keeping on toy data).
        :return: the Fragment instance that got added.
        """
        if self._contains_seq(seq):
            frag = self.get_fragment(seq)
        elif self._contains_seq(seq.revcomp_seq()):
            frag = self.get_fragment(seq.revcomp_seq())
        else:
            frag = self._create_frag(seq)

        if metadata is not None:
            frag.add_metadata(metadata)
        return frag

    def get_fragments(self) -> Iterable[Fragment]:
        return self.seq_to_frag.values()

    def get_fragment_index(self, seq: Sequence) -> int:
        return self.get_fragment(seq).index

    def get_fragment(self, seq: Sequence) -> Fragment:
        """
        Retrieves a Fragment instance corresponding to the sequence.
        Raises KeyError if the sequence is not found in the space.

        :param seq: A string.
        :return: the Fragment instance encapsulating the seq.
        """
        try:
            return self.seq_to_frag[self._seq_to_key(seq)]
        except KeyError:
            try:
                return self.seq_to_frag[self._seq_to_key(seq.revcomp_seq())]
            except KeyError:
                raise KeyError("Sequence query (len={}) not in dictionary.".format(len(seq))) from None

    def __str__(self):
        return ",".join(str(frag) for frag in self.frag_list)

    def __repr__(self):
        return ",".join(repr(frag) for frag in self.frag_list)

    def __iter__(self):
        return self.frag_list.__iter__()

    def __len__(self) -> int:
        """
        :return: The number of fragments supported by this space.
        """
        return len(self.seq_to_frag)

    def to_fasta(self, data_dir: Path) -> Path:
        from Bio import SeqIO
        out_path = data_dir / "all_fragments.fasta"
        with open(out_path, 'w') as f:
            for fragment in self:
                SeqIO.write([self.__to_fasta_record(fragment)], f, 'fasta')
        return out_path

    def fragment_files_by_length(self, out_dir: Path) -> Iterator[Tuple[int, Path]]:
        # First, assort fragments by length.
        frag_lens = sorted(set(len(f) for f in self))

        from Bio import SeqIO
        for frag_len in frag_lens:
            out_path = out_dir / f"__fragments_length_{frag_len}.fasta"
            with open(out_path, "w") as f:
                for fragment in self:
                    if len(fragment) == frag_len:
                        SeqIO.write([self.__to_fasta_record(fragment)], f, 'fasta')
            yield frag_len, out_path

    def __to_fasta_record(self, fragment: Fragment) -> SeqRecord:
        return SeqRecord(
            Seq(fragment.seq.nucleotides()),
            id="FRAGMENT_{}".format(fragment.index)
        )

    def from_fasta_record_id(self, record_id: str) -> Fragment:
        frag_idx = int(record_id[len('FRAGMENT_'):])
        return self.frag_list[frag_idx]
