from typing import Iterable, Dict
from pathlib import Path
import hashlib

from .base import Fragment, FragmentSpace
from chronostrain.util.sequences import *
from chronostrain.logging import create_logger
logger = create_logger(__name__)


class UnallocatedFragmentSpace(FragmentSpace):
    def __init__(self, fasta_path: Path):
        super().__init__()
        if not fasta_path.exists():
            fasta_path.parent.mkdir(exist_ok=True, parents=True)
            fasta_path.touch(exist_ok=False)
        self.fasta_resource = FastaIndexedResource(fasta_path)
        self._frag_dict: Dict[str, int] = {
            str(record.seq): idx
            for idx, record in enumerate(self.fasta_resource.all_records())
        }  # for fast lookups, map seq hash -> index

    @staticmethod
    def _record_id_of(record_idx: int) -> str:
        return f'FRAG_{record_idx}'

    def add_seq(self, seq: Sequence, metadata: str = None) -> Fragment:
        if isinstance(seq, DynamicFastaSequence):
            raise NotImplementedError("TODO implement cross-fasta listing.")
        elif isinstance(seq, AllocatedSequence):
            nucl = seq.nucleotides()

            if nucl not in self._frag_dict:
                record_id = f'FRAG_{len(self.fasta_resource)}'
                record_idx = self.fasta_resource.add_record(record_id, nucl)
                self._frag_dict[nucl] = record_idx
            else:
                record_idx = self._frag_dict[nucl]
                record_id = f'FRAG_{record_idx}'

            return Fragment(
                seq=DynamicFastaSequence(self.fasta_resource, record_id, known_length=len(nucl)),
                index=record_idx,
            )
        else:
            raise NotImplementedError("Unimplemented if-branch for add_seq() method.")

    def get_fragments(self) -> Iterable[Fragment]:
        yield from self.__iter__()

    def get_fragment_index(self, seq: Sequence) -> int:
        if isinstance(seq, AllocatedSequence):
            nucl = seq.nucleotides()
            if nucl not in self._frag_dict:
                raise KeyError("Sequence [{}...] is not in the fragment collection.".format(nucl[:10]))
            return self._frag_dict[nucl]
        else:
            raise NotImplementedError("Unimplemented if-branch for add_seq() method.")

    def get_fragment(self, seq: Sequence) -> Fragment:
        if isinstance(seq, DynamicFastaSequence):
            return Fragment(
                seq=seq,
                index=self.fasta_resource.index_of(seq.record_id)
            )
        elif isinstance(seq, AllocatedSequence):
            nucl = seq.nucleotides()
            if nucl not in self._frag_dict:
                raise KeyError("Sequence [{}...] is not in the fragment collection.".format(nucl[:10]))
            record_idx = self._frag_dict[nucl]
            record_id = f'FRAG_{record_idx}'
            return self.from_fasta_record_id(record_id)
        else:
            raise NotImplementedError("Unimplemented if-branch for add_seq() method.")

    def from_fasta_record_id(self, record_id: str) -> Fragment:
        return Fragment(
            seq=DynamicFastaSequence(self.fasta_resource, record_id),
            index=self.fasta_resource.index_of(record_id)
        )

    def __len__(self) -> int:
        return len(self.fasta_resource)

    def __str__(self):
        return f"UnallocatedFragments[fasta={self.fasta_resource.fasta_path}]"

    def __repr__(self):
        return f"UnallocatedFragments[fasta={self.fasta_resource.fasta_path}]"

    def __iter__(self):
        for r_idx, record in enumerate(self.fasta_resource.all_records()):
            yield Fragment(
                seq=DynamicFastaSequence(self.fasta_resource, record.id),
                index=r_idx
            )

    def to_fasta(self, data_dir: Path) -> Path:
        logger.debug("Using Unallocated/Dynamic FASTA resource; "
                     f"re-using existing fasta file {self.fasta_resource.fasta_path}.")
        return self.fasta_resource.fasta_path

    def write_fasta_records(self):
        self.fasta_resource.flush_fasta_records()
