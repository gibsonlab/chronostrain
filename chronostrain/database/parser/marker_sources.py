import re
from pathlib import Path
from typing import Tuple, Iterator, Union

from Bio import SeqIO
from Bio.SeqFeature import SeqFeature

from chronostrain.model import Marker, MarkerMetadata
from chronostrain.util.entrez import fetch_genbank, fetch_fasta
from chronostrain.util.sequences import nucleotides_to_z4, reverse_complement_seq, z4_to_nucleotides


class PrimerNotFoundError(BaseException):
    def __init__(self, seq_acc: str, forward: str, reverse: str):
        super().__init__(
            f"Primer <{forward}>--<{reverse}> not found in {seq_acc}."
        )
        self.seq_acc = seq_acc
        self.forward_primer = forward
        self.reverse_primer = reverse


class MarkerSource:
    def __init__(self, strain_id: str, seq_accession: str, marker_max_len: int, force_download: bool, data_dir: Path):
        self.strain_id = strain_id
        self.seq_accession = seq_accession
        self.marker_max_len = marker_max_len
        self.force_download = force_download
        self.data_dir = data_dir

        self._seq = None
        self._gb_record = None

    @property
    def strain_assembly_dir(self) -> Path:
        return self.data_dir / "assemblies" / self.strain_id

    @property
    def seq_nucleotide(self) -> str:
        if self._seq is None:
            if self._gb_record is not None:
                self._seq = str(self._gb_record.seq)
            else:
                fasta_path = fetch_fasta(self.seq_accession,
                                         base_dir=self.strain_assembly_dir,
                                         force_download=self.force_download)
                self._seq = str(SeqIO.read(fasta_path, "fasta").seq)
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
        result = self._regex_match_primers(forward, reverse)
        marker_seq = nucleotides_to_z4(self.seq_nucleotide[result[0]:result[1]])
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
        marker_seq = nucleotides_to_z4(self.seq_nucleotide[start_pos - 1:end_pos])
        if from_negative_strand:
            marker_seq = reverse_complement_seq(marker_seq)
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

            marker_seq = nucleotides_to_z4(str(feature.extract(self.seq_nucleotide)))
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

    # ============================= Helpers ===============================
    def _find_primer_match(self, forward_regex, reverse_regex) -> Union[None, Tuple[int, int]]:
        # noinspection PyTypeChecker
        best_hit: Tuple[int, int] = None
        best_match_len = self.marker_max_len

        forward_matches = list(re.finditer(forward_regex, self.seq_nucleotide))
        reverse_matches = list(re.finditer(reverse_regex, self.seq_nucleotide))

        for forward_match in forward_matches:
            for reverse_match in reverse_matches:
                match_length = reverse_match.end() - forward_match.start()
                if best_match_len > match_length > 0:
                    best_hit = (forward_match.start(), reverse_match.end())
                    best_match_len = match_length
        return best_hit

    def _regex_match_primers(self, forward: str, reverse: str) -> Tuple[int, int]:
        forward_primer_regex = parse_fasta_regex(forward)
        reverse_primer_regex = z4_to_nucleotides(
            reverse_complement_seq(
                nucleotides_to_z4(
                    parse_fasta_regex(reverse)
                )
            )
        )

        result = self._find_primer_match(forward_primer_regex, reverse_primer_regex)
        if result is None:
            raise PrimerNotFoundError(self.seq_accession, forward, reverse)
        return result


class CachedMarkerSource(MarkerSource):
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
        marker_seq = nucleotides_to_z4(
            str(SeqIO.read(marker_filepath, "fasta").seq)
        )
        return Marker(
            id=marker_id,
            name=marker_name,
            seq=marker_seq,
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


# ==================== Helpers
def parse_fasta_regex(sequence: str):
    fasta_translation = {
        'R': '[AG]', 'Y': '[CT]', 'K': '[GT]',
        'M': '[AC]', 'S': '[CG]', 'W': '[AT]',
        'B': '[CGT]', 'D': '[AGT]', 'H': '[ACT]',
        'V': '[ACG]', 'N': '[ACGT]', 'A': 'A',
        'C': 'C', 'G': 'G', 'T': 'T'
    }
    sequence_regex = ''
    for char in sequence:
        sequence_regex += fasta_translation[char]
    return sequence_regex
