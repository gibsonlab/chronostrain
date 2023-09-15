from pathlib import Path

from chronostrain.model import Marker, MarkerMetadata
from chronostrain.util.sequences import *
from .base import AbstractMarkerSource


class ExistingFastaMarkerSource(AbstractMarkerSource):
    def __init__(self, data_dir: Path, strain_id: str, accession: str):
        self.strain_id = strain_id

        fasta_path = self.assembly_path(data_dir, accession)
        if not fasta_path.exists():
            raise FileNotFoundError(f"Fasta file {fasta_path} not found.")
        self.fasta = FastaIndexedResource(fasta_path)

    @staticmethod
    def assembly_path(data_dir: Path, accession: str) -> Path:
        return data_dir / "assemblies" / f"{accession}.fasta"

    def extract_from_primer(self, marker_id: str, marker_name: str, forward: str, reverse: str) -> Marker:
        raise NotImplementedError("Primer-derived markers are not implemented for fasta-derived sources.")

    def extract_subseq(self,
                       marker_id: str, marker_name: str,
                       start_pos: int, end_pos: int, from_negative_strand: bool) -> Marker:
        raise NotImplementedError("Subseq-derived markers are not implemented for fasta-derived sources.")

    def extract_from_locus_tag(self, marker_id: str, marker_name: str, locus_tag: str) -> Marker:
        raise NotImplementedError("Locus tag-derived markers are not implemented for fasta-derived sources.")

    def extract_fasta_record(self, marker_id: str, marker_name: str, record_id: str, allocate: bool) -> Marker:
        if allocate:
            seq = AllocatedSequence(self.fasta.query_fasta_record(record_id))
        else:
            seq = DynamicFastaSequence(self.fasta, record_id)
        return Marker(
            id=marker_id,
            name=marker_name,
            seq=seq,
            metadata=MarkerMetadata(
                parent_strain=self.strain_id,
                parent_sequence=record_id
            )
        )
