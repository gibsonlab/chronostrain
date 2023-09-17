import pickle
from typing import Dict, List, Iterator, Iterable, Tuple
from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .fragment import Fragment
from chronostrain.util.sequences import Sequence


class FragmentSpace:
    """
    A class representing the space of fragments. Serves as a factory for Fragment instances.
    """
    def __init__(self):
        self.seq_to_frag: Dict = dict()
        self.frag_list: List[Fragment] = list()
        self.min_len: int = 100000000

    def _contains_seq(self, seq: Sequence) -> bool:
        return self._seq_to_key(seq) in self.seq_to_frag

    @staticmethod
    def _seq_to_key(seq: Sequence) -> str:
        return seq.nucleotides()

    def _create_frag(self, seq: Sequence) -> Fragment:
        frag = Fragment(index=len(self.frag_list), seq=seq)  # uid is just the index in the list.
        self.seq_to_frag[self._seq_to_key(seq)] = frag
        self.frag_list.append(frag)
        self.min_len = min(self.min_len, len(seq))
        return frag

    def add_seq(self, seq: Sequence, metadata: str = None) -> Fragment:
        """
        Tries to add a new Fragment instance encapsulating the string seq.
        If the seq is already in the space, nothing happens.

        :param seq: A string to add to the space.
        :param metadata: The fragment-specific metadata to add (useful for record-keeping on toy read_frags).
        :return: the Fragment instance that got added.
        """
        try:
            frag = self.get_fragment(seq)  # this checks for reverse complement.
        except KeyError:
            frag = self._create_frag(seq)

        if metadata is not None:
            frag.add_metadata(metadata)
        return frag

    def get_fragments(self) -> Iterable[Fragment]:
        return self.seq_to_frag.values()

    def get_fragment_by_index(self, idx: int) -> Fragment:
        return self.frag_list[idx]

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

    def __getitem__(self, frag_index: int):
        return self.get_fragment_by_index(frag_index)

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
            for frag_idx, fragment in enumerate(self):
                SeqIO.write([self.__to_fasta_record(fragment, frag_idx)], f, 'fasta')
        return out_path

    def fragment_files_by_length(self, out_dir: Path) -> Iterator[Tuple[int, Path]]:
        # First, assort fragments by length.
        frag_lens = sorted(set(len(f) for f in self))

        from Bio import SeqIO
        for frag_len in frag_lens:
            out_path = out_dir / f"__fragments_length_{frag_len}.fasta"
            with open(out_path, "w") as f:
                for frag_idx, fragment in enumerate(self):
                    if len(fragment) == frag_len:
                        SeqIO.write([self.__to_fasta_record(fragment, frag_idx)], f, 'fasta')
            yield frag_len, out_path

    def __to_fasta_record(self, fragment: Fragment, frag_idx: int, description="") -> SeqRecord:
        return SeqRecord(
            Seq(fragment.seq.nucleotides()),
            id="FRAGMENT_{}".format(frag_idx),
            description=description
        )

    def from_fasta_record_id(self, record_id: str) -> Fragment:
        frag_idx = int(record_id[len('FRAGMENT_'):])
        return self.frag_list[frag_idx]

    def save(self, path: Path):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> 'FragmentSpace':
        with open(path, 'rb') as f:
            return pickle.load(f)
