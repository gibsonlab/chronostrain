from pathlib import Path
from typing import Iterator, Tuple

from Bio import SeqIO
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord

from chronostrain.model import Marker, MarkerMetadata
from chronostrain.util.entrez import fetch_genbank, fetch_fasta
from chronostrain.util.sequences import AllocatedSequence
from .base import AbstractMarkerSource
from .helpers import regex_match_primers


class EntrezMarkerSource(AbstractMarkerSource):
    def __init__(self, strain_id: str, seq_accession: str, marker_max_len: int, force_download: bool, data_dir: Path):
        self.strain_id = strain_id
        self.seq_accession = seq_accession
        self.marker_max_len = marker_max_len
        self.force_download = force_download
        self.data_dir = data_dir

        self._seq = None
        self._gb_record = None

    @staticmethod
    def assembly_subdir(data_dir: Path, strain_id: str) -> Path:
        return data_dir / "assemblies" / strain_id

    @property
    def strain_assembly_dir(self) -> Path:
        return self.assembly_subdir(self.data_dir, self.strain_id)

    def get_fasta_record(self) -> Tuple[Path, SeqRecord]:
        fasta_path = fetch_fasta(self.seq_accession,
                                 base_dir=self.strain_assembly_dir,
                                 force_download=self.force_download)
        record = SeqIO.read(fasta_path, "fasta")
        return fasta_path, record

    @property
    def seq_nucleotide(self) -> str:
        if self._seq is None:
            if self._gb_record is not None:
                self._seq = str(self._gb_record.seq)
            else:
                _, record = self.get_fasta_record()
                self._seq = str(record.seq)
        return self._seq

    @property
    def seq_genbank_features(self) -> Iterator[SeqFeature]:
        if self._gb_record is None:
            genbank_path = fetch_genbank(self.seq_accession,
                                         base_dir=self.strain_assembly_dir,
                                         force_download=self.force_download)
            self._gb_record = list(SeqIO.parse(genbank_path, "gb"))[0]
        yield from self._gb_record.features

    def extract_from_primer(
            self,
            marker_id: str, marker_name: str, is_canonical: bool,
            forward: str, reverse: str
    ) -> Marker:
        result = regex_match_primers(self.seq_nucleotide, self.seq_accession, forward, reverse, self.marker_max_len)
        marker_seq = AllocatedSequence(self.seq_nucleotide[result[0]:result[1]])
        return Marker(
            id=marker_id,
            name=marker_name,
            seq=marker_seq,
            canonical=is_canonical,
            metadata=MarkerMetadata(
                parent_strain=self.strain_id,
                file_path=None
            )
        )

    def extract_subseq(
            self,
            marker_id: str, marker_name: str, is_canonical: bool,
            start_pos: int, end_pos: int, from_negative_strand: bool
    ) -> Marker:
        marker_seq = AllocatedSequence(self.seq_nucleotide[start_pos - 1:end_pos])
        if from_negative_strand:
            marker_seq = marker_seq.revcomp_seq()
        return Marker(
            id=marker_id,
            name=marker_name,
            seq=marker_seq,
            canonical=is_canonical,
            metadata=MarkerMetadata(
                parent_strain=self.strain_id,
                file_path=None
            )
        )

    def extract_from_locus_tag(
            self,
            marker_id: str, marker_name: str, is_canonical: bool,
            locus_tag: str
    ) -> Marker:
        for feature in self.seq_genbank_features:
            if feature.type != 'gene':
                continue
            elif 'locus_tag' not in feature.qualifiers:
                continue
            elif locus_tag != feature.qualifiers['locus_tag'][0]:
                continue

            marker_seq = AllocatedSequence(str(feature.extract(self.seq_nucleotide)))
            return Marker(
                id=marker_id,
                name=marker_name,
                seq=marker_seq,
                canonical=is_canonical,
                metadata=MarkerMetadata(
                    parent_strain=self.strain_id,
                    file_path=None
                )
            )
        raise RuntimeError(f"Genbank record `{self.seq_accession}` does not contain locus tag `{locus_tag}`.")

    def extract_fasta_record(
            self,
            marker_id: str, marker_name: str, record_id: str
    ) -> Marker:
        f_path, record = self.get_fasta_record()
        if record.id != record_id:
            raise RuntimeError("Fasta file {} does not contain the record {}.".format(
                f_path, record_id
            ))
        return Marker(
            id=marker_id,
            name=marker_name,
            seq=AllocatedSequence(str(record.seq)),
            canonical=True,
            metadata=MarkerMetadata(
                parent_strain=self.strain_id,
                file_path=f_path
            )
        )


class CachedEntrezMarkerSource(EntrezMarkerSource):
    def __init__(self, strain_id: str, data_dir: Path, seq_accession: str, marker_max_len: int, force_download: bool):
        super().__init__(strain_id, seq_accession, marker_max_len, force_download, data_dir)

    def get_marker_filepath(self, marker_id: str) -> Path:
        return (
                self.data_dir
                / "markers"
                / f"{self.strain_id}"
                / f"{marker_id}.fasta"
        )

    def save_to_disk(self, marker: Marker, target_path: Path):
        marker.metadata.file_path = target_path
        target_path.parent.mkdir(exist_ok=True, parents=True)
        SeqIO.write(
            [marker.to_seqrecord(description=f"Strain={self.strain_id},Source={self.seq_accession}")],
            target_path,
            "fasta"
        )

    def load_from_disk(self, marker_id: str, marker_name: str, is_canonical: bool, marker_filepath: Path):
        # noinspection PyBroadException
        try:
            seq = str(SeqIO.read(marker_filepath, "fasta").seq)
        except Exception as e:
            raise RuntimeError("Encountered an error while loading cached marker sequence. "
                               "The database may be corrupeted.") from e
        return Marker(
            id=marker_id,
            name=marker_name,
            seq=AllocatedSequence(seq),
            canonical=is_canonical,
            metadata=MarkerMetadata(
                parent_strain=self.strain_id,
                file_path=marker_filepath
            )
        )

    def extract_from_primer(
            self,
            marker_id: str, marker_name: str, is_canonical: bool,
            forward: str, reverse: str
    ) -> Marker:
        marker_filepath = self.get_marker_filepath(marker_id)
        if marker_filepath.exists():
            return self.load_from_disk(marker_id, marker_name, is_canonical, marker_filepath)
        else:
            marker = super().extract_from_primer(
                marker_id, marker_name, is_canonical, forward, reverse
            )
            self.save_to_disk(marker, marker_filepath)
            return marker

    def extract_subseq(
            self,
            marker_id: str, marker_name: str, is_canonical: bool,
            start_pos: int, end_pos: int, from_negative_strand: bool
    ) -> Marker:
        marker_filepath = self.get_marker_filepath(marker_id)
        if marker_filepath.exists():
            return self.load_from_disk(marker_id, marker_name, is_canonical, marker_filepath)
        else:
            marker = super().extract_subseq(
                marker_id, marker_name, is_canonical, start_pos, end_pos, from_negative_strand
            )
            self.save_to_disk(marker, marker_filepath)
            return marker

    def extract_from_locus_tag(
            self,
            marker_id: str, marker_name: str, is_canonical: bool,
            locus_tag: str
    ) -> Marker:
        marker_filepath = self.get_marker_filepath(marker_id)
        if marker_filepath.exists():
            return self.load_from_disk(marker_id, marker_name, is_canonical, marker_filepath)
        else:
            marker = super().extract_from_locus_tag(
                marker_id, marker_name, is_canonical, locus_tag
            )
            self.save_to_disk(marker, marker_filepath)
            return marker