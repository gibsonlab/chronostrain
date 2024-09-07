from pathlib import Path

from Bio import SeqIO
from chronostrain.model import Marker, MarkerMetadata
from chronostrain.util.sequences import *
from chronostrain.util.io import read_seq_file
from .base import AbstractMarkerSource


class MultiFastaMarkerSource(AbstractMarkerSource):
    def __init__(self, strain_id: str, fasta_path: Path, seq_id: str):
        self.strain_id = strain_id
        self.seq_id = seq_id

        if not fasta_path.exists():
            raise FileNotFoundError(f"[Strain ID: {strain_id}] Fasta file {fasta_path} not found.")

        self.fasta_records = [
            record.seq
            for record in read_seq_file(fasta_path, "fasta")
        ]
        # Note: this might be faster by using `samtools faidx` automatically.
        # self.fasta = FastaIndexedResource(fasta_path)
        # [example: self.fasta.query_fasta_record(record_id)]

    def extract_subseq(self,
                       record_idx: int, marker_id: str, marker_name: str,
                       start_pos: int, end_pos: int, from_negative_strand: bool) -> Marker:
        # TODO: include direct Bio.Seq.Seq -> AllocatedSequence conversion.
        record_seq = str(self.fasta_records[record_idx])
        marker_seq = AllocatedSequence(record_seq[start_pos - 1:end_pos])
        if from_negative_strand:
            marker_seq = marker_seq.revcomp_seq()
        return Marker(
            id=marker_id,
            name=marker_name,
            seq=marker_seq,
            metadata=MarkerMetadata(
                parent_strain=self.strain_id,
                parent_seq=f"{self.seq_id}${record_idx}"
            )
        )

    def extract_fasta_record(self, marker_id: str, marker_name: str, record_id: int, allocate: bool) -> Marker:
        if allocate:
            # seq = AllocatedSequence(self.fasta.query_fasta_record(record_id))
            raise NotImplementedError("todo include support for searching fasta by record ID.")
        else:
            # seq = DynamicFastaSequence(self.fasta, record_id)
            raise NotImplementedError("todo include support for memory-mapped/non-allocated sequences.")
        # return Marker(
        #     id=marker_id,
        #     name=marker_name,
        #     seq=seq,
        #     metadata=MarkerMetadata(
        #         parent_strain=self.strain_id,
        #         parent_seq=f"{self.seq_id}${record_idx}"
        #     )
        # )
