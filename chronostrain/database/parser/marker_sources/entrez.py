from pathlib import Path
from typing import Iterator, Tuple, Union
import re

from Bio import SeqIO
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord

from chronostrain.model import Marker, MarkerMetadata
from chronostrain.util.entrez import fetch_genbank, fetch_fasta
from chronostrain.util.sequences import AllocatedSequence
from .base import AbstractMarkerSource


class EntrezMarkerSource(AbstractMarkerSource):
    def __init__(self, strain_id: str, seq_id: str, seq_path: Union[Path, None], marker_max_len: int, force_download: bool, data_dir: Path):
        self.strain_id = strain_id
        self.seq_id = seq_id
        self.seq_path = seq_path
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
        if self.seq_path is not None:
            fasta_path = self.seq_path
        else:
            # Default behavior: attempt to download if file location is not specified.
            fasta_path = fetch_fasta(self.seq_id,
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
            genbank_path = fetch_genbank(self.seq_id,
                                         base_dir=self.strain_assembly_dir,
                                         force_download=self.force_download)
            self._gb_record = list(SeqIO.parse(genbank_path, "gb"))[0]
        yield from self._gb_record.features

    def extract_subseq(
            self,
            record_idx: int,
            marker_id: str, marker_name: str,
            start_pos: int, end_pos: int, from_negative_strand: bool
    ) -> Marker:
        # here, record_idx is unused; the FASTA file is assumed to have exactly one sequence.
        marker_seq = AllocatedSequence(self.seq_nucleotide[start_pos - 1:end_pos])
        if from_negative_strand:
            marker_seq = marker_seq.revcomp_seq()
        return Marker(
            id=marker_id,
            name=marker_name,
            seq=marker_seq,
            metadata=MarkerMetadata(
                parent_strain=self.strain_id,
                parent_seq=self.seq_id,
            )
        )

    def extract_from_locus_tag(
            self,
            marker_id: str, marker_name: str,
            locus_tag: str
    ) -> Marker:
        """
        Deprecated; kept for legacy/posterity reasons.
        :param marker_id: the target marker ID to set for the Marker object.
        :param marker_name: the target marker name to set for the Marker object.
        :param locus_tag: the locus tag ID to look for in the genbank (gff) file.
        :return:
        """
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
                metadata=MarkerMetadata(
                    parent_strain=self.strain_id,
                    parent_seq=self.seq_id,
                )
            )
        raise RuntimeError(f"Genbank record `{self.seq_id}` does not contain locus tag `{locus_tag}`.")

    def extract_fasta_record(
            self,
            marker_id: str, marker_name: str, record_id: str, allocate: bool
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
            metadata=MarkerMetadata(
                parent_strain=self.strain_id,
                parent_seq=self.seq_id,
            )
        )


class CachedEntrezMarkerSource(EntrezMarkerSource):
    def __init__(self,
                 strain_id: str,
                 data_dir: Path,
                 seq_id: str,
                 seq_path: Union[Path, None],
                 marker_max_len: int,
                 force_download: bool):
        super().__init__(strain_id, seq_id, seq_path, marker_max_len, force_download, data_dir)

    def get_marker_filepath(self, marker_id: str) -> Path:
        marker_id_for_filename = re.sub(r'[^\w\s]', '_', marker_id)
        return (
                self.data_dir
                / "markers"
                / f"{self.strain_id}"
                / f"{marker_id_for_filename}.fasta"
        )

    def save_to_disk(self, marker: Marker, target_path: Path):
        marker.metadata.file_path = target_path
        target_path.parent.mkdir(exist_ok=True, parents=True)
        SeqIO.write(
            [marker.to_seqrecord(description=f"Strain={self.strain_id},Source={self.seq_id}")],
            target_path,
            "fasta"
        )

    def load_from_disk(self, marker_id: str, marker_name: str, marker_filepath: Path):
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
            metadata=MarkerMetadata(
                parent_strain=self.strain_id,
                parent_seq=self.seq_id
            )
        )

    def extract_subseq(
            self,
            record_idx: int,
            marker_id: str, marker_name: str,
            start_pos: int, end_pos: int, from_negative_strand: bool
    ) -> Marker:
        marker_filepath = self.get_marker_filepath(marker_id)
        if marker_filepath.exists():
            return self.load_from_disk(marker_id, marker_name, marker_filepath)
        else:
            marker = super().extract_subseq(
                record_idx, marker_id, marker_name, start_pos, end_pos, from_negative_strand
            )
            self.save_to_disk(marker, marker_filepath)
            return marker

    def extract_from_locus_tag(
            self,
            marker_id: str, marker_name: str,
            locus_tag: str
    ) -> Marker:
        marker_filepath = self.get_marker_filepath(marker_id)
        if marker_filepath.exists():
            return self.load_from_disk(marker_id, marker_name, marker_filepath)
        else:
            marker = super().extract_from_locus_tag(
                marker_id, marker_name, locus_tag
            )
            self.save_to_disk(marker, marker_filepath)
            return marker
