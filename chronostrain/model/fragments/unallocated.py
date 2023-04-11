from typing import Iterable, Set
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

    @staticmethod
    def nucleotide_key(seq: str) -> str:
        # Dynamically generate an ID from a non-fasta record. For now, use md5-based hashing.
        return hashlib.md5(seq).hexdigest()

    def contains_seq(self, seq: Sequence) -> bool:
        if isinstance(seq, DynamicFastaSequence):
            return self.fasta_resource.contains(seq.record_id)
        elif isinstance(seq, AllocatedSequence):
            return self.fasta_resource.contains(self.nucleotide_key(seq.nucleotides()))
        else:
            raise NotImplementedError("Unimplemented if-branch for add_seq() method.")

    def add_seq(self, seq: Sequence, metadata: str = None) -> Fragment:
        if isinstance(seq, DynamicFastaSequence):
            raise NotImplementedError("TODO implement cross-fasta listing.")
        elif isinstance(seq, AllocatedSequence):
            nucl = seq.nucleotides()
            record_id = self.nucleotide_key(nucl)
            try:
                record_idx = self.fasta_resource.index_of(record_id)
            except FastaRecordNotFound:
                record_idx = self.fasta_resource.add_record(record_id, nucl)

            return Fragment(
                seq=DynamicFastaSequence(self.fasta_resource, record_id),
                index=record_idx,
            )
        else:
            raise NotImplementedError("Unimplemented if-branch for add_seq() method.")

    def get_fragments(self) -> Iterable[Fragment]:
        yield from self.__iter__()

    def get_fragment(self, seq: Sequence) -> Fragment:
        if isinstance(seq, DynamicFastaSequence):
            return Fragment(
                seq=seq,
                index=self.fasta_resource.index_of(seq.record_id)
            )
        elif isinstance(seq, AllocatedSequence):
            nucl = seq.nucleotides()
            record_id = self.nucleotide_key(nucl)
            return self.from_fasta_record_id(record_id)
        else:
            raise NotImplementedError("Unimplemented if-branch for add_seq() method.")

    def from_fasta_record_id(self, record_id: str) -> Fragment:
        return Fragment(
            seq=DynamicFastaSequence(self.fasta_resource, record_id),
            index=self.fasta_resource.index_of(record_id)
        )

    def __len__(self) -> int:
        return self.fasta_resource.num_records

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
