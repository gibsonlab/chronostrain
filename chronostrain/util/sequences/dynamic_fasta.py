from pathlib import Path
from typing import Iterator, Dict
import io
from Bio import SeqIO
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .base import Sequence
from .z4 import nucleotides_to_z4
from ..external.samtools import faidx


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
    def __init__(self, fasta_path: Path, use_gzip: bool = False):
        self.fasta_path = fasta_path
        self.use_gzip = use_gzip
        self.create_index()

        self._id_indices: Dict[str, int] = {}

    @property
    def num_records(self):
        # Lazy loading
        if len(self._id_indices) == 0:
            with open(self.fasta_path, "rt") as f:
                self._id_indices = {
                    record.id: r_idx
                    for r_idx, record in enumerate(SeqIO.parse(f, "fasta"))
                }
        return len(self._id_indices)

    def create_index(self):
        faidx(fasta_path=self.fasta_path, silent=False)

    def all_records(self) -> Iterator[SeqRecord]:
        with open(self.fasta_path, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                yield record

    def contains(self, record_id):
        try:
            self.query_fasta_record(record_id)
            return True
        except FastaRecordNotFound:
            return False

    def query_fasta_record(self, record_id: str) -> str:
        buf = io.StringIO()
        try:
            faidx(
                fasta_path=self.fasta_path,
                query_regions=[record_id],
                buf=buf,
                silent=True
            )
            buf.seek(0)

            record = SeqIO.read(buf, "fasta")
            if str(record.seq).startswith("[W::fai_fetch]") or str(record.seq).endswith("empty sequence"):
                raise FastaRecordNotFound(self.fasta_path, record_id)

            return str(record.seq)
        finally:
            buf.close()

    def add_record(self, record_id: str, nucleotides: str) -> int:
        with open(self.fasta_path, 'a') as f:
            record = SeqRecord(id=record_id, seq=Seq(nucleotides))
            SeqIO.write([record], f, format='fasta')
            new_idx = self.num_records
            self._id_indices[record_id] = new_idx
        return new_idx

    def index_of(self, record_id: str) -> int:
        return self._id_indices[record_id]


class DynamicFastaSequence(Sequence):
    """
    Represents a single sequence, whose contents are not pre-allocated in memory but rather loaded via a FASTA
    record accessed using an index amenable to random querying.
    """
    def __init__(self, seq_resource: FastaIndexedResource, record_id: str):
        self.seq_resource = seq_resource
        self.record_id = record_id
        self._len = -1

    def nucleotides(self) -> str:
        return self.seq_resource.query_fasta_record(self.record_id)

    def bytes(self) -> np.ndarray:
        return nucleotides_to_z4(self.seq_resource.query_fasta_record(self.record_id))

    def revcomp_seq(self) -> 'Sequence':
        pass

    def __len__(self):
        if self._len < 0:  # Lazy loading
            self._len = len(self.nucleotides())
        return self._len

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Dynamic[{self.seq_resource.fasta_path}:{self.record_id}]"
