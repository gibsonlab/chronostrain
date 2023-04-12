from pathlib import Path
from typing import Iterator, List
import io
from Bio import SeqIO
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .base import Sequence
from .z4 import nucleotides_to_z4
from ..external.cdbtools import cdbfasta, cdbyank


class FastaRecordNotFound(BaseException):
    def __init__(self, fasta_path: Path, record_id: str):
        super().__init__("Unable to find record `{}` in fasta file {}.".format(
            record_id, fasta_path
        ))
        self.fasta_path = fasta_path
        self.record_id = record_id


class FastaIndexedResource(object):
    """
    Fasta sequence resource, where querying is implemented via `samtools faidx`.
    Alternatively, could use python-native faidx (see https://github.com/mdshw5/pyfaidx/), whose pros/cons
    should be explored in the future.
    """
    _TEMP_CAPACITY = 50000

    def __init__(self, fasta_path: Path, use_gzip: bool = False):
        self.fasta_path = fasta_path
        self.index_path = fasta_path.with_name(f'{fasta_path.name}.cidx')
        self.use_gzip = use_gzip
        with open(self.fasta_path, "rt") as f:
            self._id_indices = {
                record.id: r_idx
                for r_idx, record in enumerate(SeqIO.parse(f, "fasta"))
            }

            self._new_records: List[SeqRecord] = []

    def __len__(self):
        return len(self._id_indices)

    def create_index(self, silent: bool = False):
        cdbfasta(fasta_path=self.fasta_path, silent=silent)

    def all_records(self) -> Iterator[SeqRecord]:
        self.flush_fasta_records()
        with open(self.fasta_path, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                yield record

    def contains(self, record_id):
        if len(self) == 0:
            return False

        try:
            self.query_fasta_record(record_id)
            return True
        except FastaRecordNotFound:
            return False

    def query_fasta_record(self, record_id: str, revcomp: bool = False) -> str:
        self.flush_fasta_records()

        buf = io.StringIO()
        try:
            cdbyank(
                index_path=self.index_path,
                target_accession=record_id,
                buf=buf,
                silent=True
            )
            buf.seek(0)

            record: SeqRecord = SeqIO.read(buf, "fasta")
            if str(record.seq).startswith("[W::fai_fetch]") or str(record.seq).endswith("empty sequence"):
                raise FastaRecordNotFound(self.fasta_path, record_id)

            if revcomp:
                return str(record.seq.reverse_complement())
            else:
                return str(record.seq)
        finally:
            buf.close()

    def add_record(self, record_id: str, nucleotides: str) -> int:
        new_record = SeqRecord(id=record_id, seq=Seq(nucleotides), description="")
        new_idx = len(self)
        self._id_indices[record_id] = new_idx
        self._new_records.append(new_record)
        if len(self._new_records) >= FastaIndexedResource._TEMP_CAPACITY:
            self.flush_fasta_records()
        return new_idx

    def flush_fasta_records(self):
        """
        Called automatically upon querying of sequences, or periodically after adding a certain threshold of records.
        Should also be called manually after a batch of add_record() calls.
        """
        if len(self._new_records) == 0:
            return

        with open(self.fasta_path, 'a') as f:
            SeqIO.write(self._new_records, f, format='fasta')
            self._new_records = []
        self.create_index(silent=False)

    def index_of(self, record_id: str) -> int:
        if record_id not in self._id_indices:
            raise FastaRecordNotFound(self.fasta_path, record_id)
        return self._id_indices[record_id]


class DynamicFastaSequence(Sequence):
    """
    Represents a single sequence, whose contents are not pre-allocated in memory but rather loaded via a FASTA
    record accessed using an index amenable to random querying.
    """
    def __init__(self, seq_resource: FastaIndexedResource, record_id: str, is_reverse_complement: bool = False, known_length: int = -1):
        self.seq_resource = seq_resource
        self.record_id = record_id
        self.is_reverse_complement = is_reverse_complement
        self._len = known_length

    def nucleotides(self) -> str:
        return self.seq_resource.query_fasta_record(self.record_id, revcomp=self.is_reverse_complement)

    def bytes(self) -> np.ndarray:
        return nucleotides_to_z4(self.nucleotides())

    def revcomp_seq(self) -> 'Sequence':
        new_seq = DynamicFastaSequence(self.seq_resource, self.record_id, not self.is_reverse_complement)
        if self._len > 0:
            new_seq._len = self._len
        return new_seq

    def __len__(self):
        if self._len < 0:  # Lazy loading
            self._len = len(self.nucleotides())
        return self._len

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Dynamic[{self.seq_resource.fasta_path}:{self.record_id}]"
