import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Union, Tuple

from Bio import SeqIO

from chronostrain.config import cfg
from chronostrain.model import Strain, Marker, MarkerMetadata, StrainMetadata
from chronostrain.util.entrez import fetch_genbank
from chronostrain.util.sequences import complement_seq, nucleotides_to_z4

from .base import AbstractDatabaseParser, StrainDatabaseParseError


from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


# =====================================================================
# JSON entry dataclasses. Each class implements a deserialize() method.
# =====================================================================


@dataclass
class StrainEntry:
    genus: str
    species: str
    accession: str
    marker_entries: List["MarkerEntry"]
    index: int

    def __str__(self):
        return "(Strain Entry #{index}: {genus} {species})".format(
            index=self.index,
            genus=self.genus,
            species=self.species
        )

    def __repr__(self):
        return "{}(idx={},acc={})".format(
            self.__class__.__name__,
            self.index,
            self.accession
        )

    @staticmethod
    def deserialize(json_dict: dict, idx: int):
        try:
            genus = json_dict["genus"]
        except KeyError:
            raise StrainDatabaseParseError("Missing entry `genus` from json entry.")

        try:
            species = json_dict["species"]
        except KeyError:
            raise StrainDatabaseParseError("Missing entry `species` from json entry.")

        try:
            accession = json_dict["accession"]
        except KeyError:
            raise StrainDatabaseParseError("Missing entry `accession` from json entry.")

        try:
            markers_arr = (json_dict["markers"])
        except KeyError:
            raise StrainDatabaseParseError("Missing entry `markers` from json entry.")

        marker_entries = []
        entry = StrainEntry(genus=genus,
                            species=species,
                            accession=accession,
                            marker_entries=marker_entries,
                            index=idx)
        for idx, marker_dict in enumerate(markers_arr):
            marker_entries.append(MarkerEntry.deserialize(marker_dict, idx, entry))
        return entry


@dataclass
class MarkerEntry:
    name: str
    index: int
    parent: StrainEntry

    @staticmethod
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> "MarkerEntry":
        marker_type = entry_dict['type']
        if marker_type == 'tag':
            return TagMarkerEntry.deserialize(entry_dict, idx, parent)
        elif marker_type == 'primer':
            return PrimerMarkerEntry.deserialize(entry_dict, idx, parent)
        else:
            raise StrainDatabaseParseError("Unexpected type `{}` in marker entry {} of {}".format(marker_type, idx, parent))


@dataclass
class TagMarkerEntry(MarkerEntry):
    locus_tag: str

    def __str__(self):
        return "(Tag Marker Entry #{} of {}: {})".format(
            self.index,
            self.parent.accession,
            self.name
        )

    def __repr__(self):
        return "{}(parent={},idx={},locus_tag={})".format(
            self.__class__.__name__,
            self.parent.accession,
            self.index,
            self.locus_tag
        )

    @staticmethod
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> "TagMarkerEntry":
        return TagMarkerEntry(name=entry_dict['name'],
                              index=idx,
                              locus_tag=entry_dict['locus_tag'],
                              parent=parent)


@dataclass
class PrimerMarkerEntry(MarkerEntry):
    forward: str
    reverse: str

    def __str__(self):
        return "(Primer Marker Entry #{} of {}: {}-{})".format(
            self.index,
            self.parent.accession,
            self.forward[:4] + "..." if len(self.forward) > 4 else self.forward,
            self.reverse[:4] + "..." if len(self.reverse) > 4 else self.reverse,
        )

    def __repr__(self):
        return "{}(parent={},idx={},fwd={},rev={})".format(
            self.__class__.__name__,
            self.parent.accession,
            self.index,
            self.forward,
            self.reverse
        )

    def entry_id(self) -> str:
        return "{}[{}]".format(
            self.parent.accession,
            '-'.join([self.forward, self.reverse])
        )


    @staticmethod
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> "PrimerMarkerEntry":
        return PrimerMarkerEntry(name=entry_dict['name'],
                                 index=idx,
                                 forward=entry_dict['forward'],
                                 reverse=entry_dict['reverse'],
                                 parent=parent)


# =================================================================
#  Sequence parsers. Operates on strings (nucleotide sequences)
#  to extract desired subsequences.
# =================================================================

class NucleotideSubsequence:
    def __init__(self, name: str, id: str, start_index: int, end_index: int, complement: bool):
        self.name = name
        self.id = id
        self.start_index = start_index
        self.end_index = end_index
        self.complement = complement

    def get_subsequence(self, nucleotides: str) -> str:
        # Note: The "complement" feature is not used; genbank annotation gives the resulting protein in terms
        # of the translation (on either forward or reverse strand), but we are modeling shotgun reads
        # (nucleotide substrings), not expression/protein levels.
        return nucleotides[self.start_index:self.end_index]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{name}[{locus}:{start}:{end}]".format(
            name=self.name,
            locus=self.id,
            start=self.start_index,
            end=self.end_index
        )


class SubsequenceLoader:
    """
    A class designed to extract specific marker subsequences from NCBI, by pulling substrings from NCBI's database.
    """

    def __init__(self,
                 strain_accession: str,
                 genbank_filename: Path,
                 marker_entries: List[MarkerEntry],
                 marker_max_len: int):
        self.strain_accession = strain_accession
        self.genbank_filename = genbank_filename

        self.tag_entries: List[TagMarkerEntry] = []
        self.primer_entries: List[PrimerMarkerEntry] = []
        for entry in marker_entries:
            if isinstance(entry, TagMarkerEntry):
                self.tag_entries.append(entry)
            elif isinstance(entry, PrimerMarkerEntry):
                self.primer_entries.append(entry)
            else:
                raise NotImplementedError("Entry class `{}` not implemented.".format(entry.__class__.__name__))

        self.full_genome = None  # Lazy loading in get_full_genome()
        self.marker_max_len = int(marker_max_len)

    def get_full_genome(self, trim_debug=None) -> str:
        if self.full_genome is None:
            record = next(SeqIO.parse(self.genbank_filename, "genbank"))
            self.full_genome = str(record.seq)
            if trim_debug is not None:
                self.full_genome = self.full_genome[:trim_debug]
        return self.full_genome

    def marker_filepath(self, marker_id: str) -> Path:
        return (
                Path(cfg.database_cfg.data_dir)
                / "{acc}-{marker}.fasta".format(acc=self.strain_accession, marker=marker_id)
        )

    def parse_markers(self, force_refresh: bool = False) -> Iterator[Marker]:
        """
        Markers are expected to be a list of JSON objects of one of the following formats:
        (1) {'type': 'tag', 'name': <COMMON_NAME>, 'id': <NCBI_ID>}
        (2) {'type': 'primer', 'name': <COMMON_NAME>, 'forward': <FORWARD_SEQ>, 'reverse': <REV_SEQ>}

        This method parses either type, depending on the 'type' field.
        :return: A list of marker instances.
        """

        if not force_refresh:
            for marker in self.load_entries_from_disk():
                yield marker

        for subseq_obj in itertools.chain(self.get_subsequences_from_tags(), self.get_subsequences_from_primers()):
            marker_filepath = self.marker_filepath(subseq_obj.id)
            marker = Marker(
                name=subseq_obj.name,
                id=subseq_obj.id,
                seq=nucleotides_to_z4(subseq_obj.get_subsequence(self.get_full_genome())),
                metadata=MarkerMetadata(
                    parent_accession=self.strain_accession,
                    file_path=marker_filepath
                )
            )
            self.save_marker_to_disk(marker, marker_filepath)
            yield marker

    def load_entries_from_disk(self) -> Iterator[Marker]:
        tag_entries_missed = []
        for tag_entry in self.tag_entries:
            marker_name = tag_entry.name
            marker_id = tag_entry.locus_tag
            marker_filepath = self.marker_filepath(marker_id)
            try:
                yield self.load_marker_from_disk(marker_filepath, marker_name, marker_id)
            except FileNotFoundError:
                tag_entries_missed.append(tag_entry)
            except StrainDatabaseParseError as e:
                logger.warning(str(e))
                tag_entries_missed.append(tag_entry)
        self.tag_entries = tag_entries_missed

        primer_entries_missed = []
        for primer_entry in self.primer_entries:
            marker_name = primer_entry.name
            marker_id = primer_entry.entry_id()
            marker_filepath = self.marker_filepath(marker_id)
            try:
                yield self.load_marker_from_disk(marker_filepath, marker_name, marker_id)
            except FileNotFoundError:
                primer_entries_missed.append(primer_entry)
            except StrainDatabaseParseError as e:
                logger.warning(str(e))
                primer_entries_missed.append(primer_entry)
        self.primer_entries = primer_entries_missed

    def save_marker_to_disk(self, marker: Marker, filepath: Path):
        SeqIO.write([
            marker.to_seqrecord(description="Strain:{}".format(self.strain_accession))
        ], filepath, "fasta")

    def load_marker_from_disk(self, filepath: Path, expected_marker_name: str, expected_marker_id: str) -> Marker:
        record = next(SeqIO.parse(filepath, "fasta"))
        accession_token, name_token, id_token = record.id.split("|")
        if accession_token != self.strain_accession:
            raise StrainDatabaseParseError(
                "Marker's strain accession {} does not match input strain accession {}. (File={})".format(
                    accession_token,
                    self.strain_accession,
                    filepath
                )
            )
        elif name_token != expected_marker_name:
            raise StrainDatabaseParseError("Marker name {} does not match input name {}. (File={})".format(
                name_token,
                expected_marker_name,
                filepath
            ))
        elif id_token != expected_marker_id:
            raise StrainDatabaseParseError("Marker id {} does not match input id {}. (File={})".format(
                id_token,
                expected_marker_id,
                filepath
            ))
        else:
            return Marker(
                name=expected_marker_name,
                id=expected_marker_id,
                seq=nucleotides_to_z4(str(record.seq)),
                metadata=MarkerMetadata(
                    parent_accession=self.strain_accession,
                    file_path=filepath
                )
            )

    def get_subsequences_from_tags(self) -> List[NucleotideSubsequence]:
        if len(self.tag_entries) == 0:
            return []

        tags_to_entries = {
            entry.locus_tag: entry
            for entry in self.tag_entries
        }

        tags_found = set()

        subsequences = []

        for record in SeqIO.parse(self.genbank_filename, format="genbank"):
            for feature in record.features:
                if feature.type == "gene" and "locus_tag" in feature.qualifiers:
                    locus_tag = feature.qualifiers["locus_tag"][0]
                    if locus_tag in tags_to_entries:
                        # Found a match for the specified locus_tag.
                        tags_found.add(locus_tag)
                        tag_entry = tags_to_entries[locus_tag]
                        loc = feature.location

                        subsequences.append(NucleotideSubsequence(
                            name=tag_entry.name,
                            id=tag_entry.locus_tag,
                            start_index=loc.start,
                            end_index=loc.end,
                            complement=loc.strand == -1
                        ))

        for tag, entry in tags_to_entries.items():
            if tag not in tags_found:
                logger.warning("Unable to find matches for tag entry {}.".format(entry))

        return subsequences

    def get_subsequences_from_primers(self) -> List[NucleotideSubsequence]:
        subsequences = []

        for entry in self.primer_entries:
            logger.debug("Performing regex search for primer {}-{}.".format(entry.forward, entry.reverse))
            '''
            Try both possibilities. (forward/reverse both in 5' -> 3', versus both 3' -> 5')
            '''
            result = self._regex_match_primers(entry.forward, entry.reverse)
            if result is None:
                result = self._regex_match_primers(entry.reverse, entry.forward)

            if result is None:
                logger.warning("Couldn't find matches for primer entry {}.".format(
                    str(entry)
                ))
            else:
                logger.debug("Found primer match: ({},{})".format(result[0], result[1]))
                subsequences.append(NucleotideSubsequence(
                    name=entry.name,
                    id=entry.entry_id(),
                    start_index=result[0],
                    end_index=result[1],
                    complement=False
                ))

        return subsequences

    def _regex_match_primers(self, forward: str, reverse: str) -> Union[None, Tuple[int, int]]:
        forward_primer_regex = self.parse_fasta_regex(forward)
        reverse_primer_regex = complement_seq(self.parse_fasta_regex(reverse[::-1]), ignore_keyerror=True)
        return self.find_primer_match(forward_primer_regex, reverse_primer_regex)

    @staticmethod
    def parse_fasta_regex(sequence):
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

    def find_primer_match(self, forward_regex, reverse_regex) -> Union[None, Tuple[int, int]]:
        best_match = (None, self.marker_max_len)
        forward_matches = list(re.finditer(forward_regex, self.get_full_genome()))
        reverse_matches = list(re.finditer(reverse_regex, self.get_full_genome()))

        for forward_match in forward_matches:
            for reverse_match in reverse_matches:
                match_length = reverse_match.end() - forward_match.start()
                if best_match[1] > match_length > 0:
                    best_match = ((forward_match.start(), reverse_match.end()), match_length)
        return best_match[0]


class JSONParser(AbstractDatabaseParser):
    def __init__(self, entries_file: Path, marker_max_len: int, force_refresh: bool = False):
        self.entries_file = entries_file
        self.marker_max_len = marker_max_len
        self.force_refresh = force_refresh

    def strains(self) -> Iterator[Strain]:
        logger.info("Loading from JSON marker database file {}.".format(self.entries_file))
        logger.debug("Data will be saved to/load from: {}".format(cfg.database_cfg.data_dir))
        for strain_entry in self.strain_entries():
            genbank_filename = fetch_genbank(strain_entry.accession,
                                             base_dir=cfg.database_cfg.data_dir,
                                             force_download=self.force_refresh)

            sequence_loader = SubsequenceLoader(
                strain_accession=strain_entry.accession,
                genbank_filename=genbank_filename,
                marker_entries=strain_entry.marker_entries,
                marker_max_len=self.marker_max_len
            )

            strain_markers = [
                marker for
                marker in sequence_loader.parse_markers(force_refresh=self.force_refresh)
            ]

            if len(strain_markers) == 0:
                logger.warning("No markers parsed for entry {}.".format(strain_entry))
            else:
                logger.debug("Strain {} loaded with {} markers.".format(
                    strain_entry.accession,
                    len(strain_markers)
                ))

            yield Strain(
                id=strain_entry.accession,
                markers=strain_markers,
                metadata=StrainMetadata(
                    ncbi_accession=strain_entry.accession,
                    genus=strain_entry.genus,
                    species=strain_entry.species,
                    file_path=genbank_filename
                )
            )

    def strain_entries(self) -> Iterator[StrainEntry]:
        """
        Deserialize JSON into StrainEntry instances.
        """
        with open(self.entries_file, "r") as f:
            for idx, strain_dict in enumerate(json.load(f)):
                yield StrainEntry.deserialize(strain_dict, idx)
