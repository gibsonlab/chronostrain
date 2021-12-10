import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Union, Tuple

from Bio import SeqIO
from chronostrain.util.entrez import fetch_fasta

from chronostrain.config import cfg
from chronostrain.model import Strain, Marker, MarkerMetadata, StrainMetadata
from chronostrain.util.entrez import fetch_genbank
from chronostrain.util.sequences import *

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
    source: str
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
            raise StrainDatabaseParseError("Missing entry `genus` from json strain entry, index {}.".format(idx))

        try:
            species = json_dict["species"]
        except KeyError:
            raise StrainDatabaseParseError("Missing entry `species` from json strain entry, index {}.".format(idx))

        try:
            accession = json_dict["accession"]
        except KeyError:
            raise StrainDatabaseParseError("Missing entry `accession` from json strain entry, index {}.".format(idx))

        try:
            markers_arr = json_dict["markers"]
        except KeyError:
            raise StrainDatabaseParseError("Missing entry `markers` from json strain entry, index {}.".format(idx))

        try:
            source = json_dict['source']
        except KeyError:
            raise StrainDatabaseParseError("Missing entry `source` from json strain entry, index {}.".format(idx))

        marker_entries = []
        entry = StrainEntry(genus=genus,
                            species=species,
                            accession=accession,
                            marker_entries=marker_entries,
                            source=source,
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
        if 'type' not in entry_dict:
            raise StrainDatabaseParseError(
                "Missing entry `type` from json marker entry of strain entry {}.".format(parent.accession)
            )

        marker_type = entry_dict['type']
        if marker_type == 'tag':
            return TagMarkerEntry.deserialize(entry_dict, idx, parent)
        elif marker_type == 'primer':
            return PrimerMarkerEntry.deserialize(entry_dict, idx, parent)
        elif marker_type == 'subseq':
            return SubseqMarkerEntry.deserialize(entry_dict, idx, parent)
        else:
            raise StrainDatabaseParseError("Unexpected type `{}` in marker entry {} of {}".format(
                marker_type, idx, parent
            ))


@dataclass
class TagMarkerEntry(MarkerEntry):
    locus_tag: str
    is_canonical: bool

    def __str__(self):
        return "(Tag Marker Entry #{} of {}: {}{})".format(
            self.index,
            self.parent.accession,
            self.name,
            " <Canonical>" if self.is_canonical else ""
        )

    def __repr__(self):
        return "{}(parent={},idx={},locus_tag={},canonical={})".format(
            self.__class__.__name__,
            self.parent.accession,
            self.index,
            self.locus_tag,
            self.is_canonical
        )

    @staticmethod
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> "TagMarkerEntry":
        if 'name' not in entry_dict:
            raise StrainDatabaseParseError(
                "Missing entry `name` from json marker entry of strain entry {}.".format(parent.accession)
            )
        if 'locus_tag' not in entry_dict:
            raise StrainDatabaseParseError(
                "Missing entry `locus_tag` from json marker entry of strain entry {}.".format(parent.accession)
            )

        return TagMarkerEntry(
            name=entry_dict['name'],
            index=idx,
            locus_tag=entry_dict['locus_tag'],
            parent=parent,
            is_canonical=('canonical' in entry_dict) and (str(entry_dict['canonical']).strip().lower() == "true"),
        )


@dataclass
class PrimerMarkerEntry(MarkerEntry):
    forward: str
    reverse: str
    is_canonical: bool

    def __str__(self):
        return "(Primer Marker Entry #{} of {}: {}-{}{})".format(
            self.index,
            self.parent.accession,
            self.forward[:4] + "..." if len(self.forward) > 4 else self.forward,
            self.reverse[:4] + "..." if len(self.reverse) > 4 else self.reverse,
            " <Canonical>" if self.is_canonical else ""
        )

    def __repr__(self):
        return "{}(parent={},idx={},fwd={},rev={},Canonical={})".format(
            self.__class__.__name__,
            self.parent.accession,
            self.index,
            self.forward,
            self.reverse,
            self.is_canonical
        )

    def entry_id(self) -> str:
        return "{}[Primer:{}]".format(
            self.parent.accession,
            '-'.join([self.forward, self.reverse])
        )

    @staticmethod
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> "PrimerMarkerEntry":
        if 'name' not in entry_dict:
            raise StrainDatabaseParseError(
                "Missing entry `name` from json marker entry of strain entry {}.".format(parent.accession)
            )
        if 'forward' not in entry_dict:
            raise StrainDatabaseParseError(
                "Missing entry `forward` from json marker entry of strain entry {}.".format(parent.accession)
            )
        if 'reverse' not in entry_dict:
            raise StrainDatabaseParseError(
                "Missing entry `reverse` from json marker entry of strain entry {}.".format(parent.accession)
            )

        return PrimerMarkerEntry(
            name=entry_dict['name'],
            index=idx,
            forward=entry_dict['forward'],
            reverse=entry_dict['reverse'],
            parent=parent,
            is_canonical=('canonical' in entry_dict) and (str(entry_dict['canonical']).strip().lower() == "true"),
        )


@dataclass
class SubseqMarkerEntry(MarkerEntry):
    start_pos: int
    end_pos: int
    is_canonical: bool
    is_negative_strand: bool
    id: str

    def __str__(self):
        return "(Subseq Marker Entry #{} of {}: {}-{}{})".format(
            self.index,
            self.parent.accession,
            self.start_pos,
            self.end_pos,
            " <Canonical>" if self.is_canonical else ""
        )

    def __repr__(self):
        return "{}(parent={},idx={},start={},end={},Canonical={})".format(
            self.__class__.__name__,
            self.parent.accession,
            self.index,
            self.start_pos,
            self.end_pos,
            self.is_canonical
        )

    @staticmethod
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> "SubseqMarkerEntry":
        if 'name' not in entry_dict:
            raise StrainDatabaseParseError(
                f"Missing entry `name` from json marker entry of strain entry {parent.accession}."
            )
        if 'start' not in entry_dict:
            raise StrainDatabaseParseError(
                f"Missing entry `start` from json marker entry of strain entry {parent.accession}."
            )
        if 'end' not in entry_dict:
            raise StrainDatabaseParseError(
                f"Missing entry `end` from json marker entry of strain entry {parent.accession}."
            )
        if 'strand' not in entry_dict:
            raise StrainDatabaseParseError(
                f"Missing entry `strand` from json marker entry of strain entry {parent.accession}."
            )
        if 'id' not in entry_dict:
            raise StrainDatabaseParseError(
                f"Missing entry `id` from json marker entry of strain entry {parent.accession}."
            )

        strand_str = entry_dict['strand']
        if strand_str == '+':
            is_negative_strand = False
        elif strand_str == '-':
            is_negative_strand = True
        else:
            raise ValueError(
                f"Unrecognizable value `{strand_str}` of entry `strand` in {parent.accession}."
            )

        return SubseqMarkerEntry(
            name=entry_dict['name'],
            index=idx,
            start_pos=entry_dict['start'],
            end_pos=entry_dict['end'],
            is_negative_strand=is_negative_strand,
            parent=parent,
            is_canonical=('canonical' in entry_dict) and (str(entry_dict['canonical']).strip().lower() == "true"),
            id=entry_dict['id']
        )


# =================================================================
#  Sequence parsers. Operates on strings (nucleotide sequences)
#  to extract desired subsequences.
# =================================================================

class NucleotideSubsequence:
    def __init__(self, name: str, id: str, start_index: int, end_index: int, complement: bool, is_canonical: bool):
        self.name = name
        self.id = id
        self.start_index = start_index
        self.end_index = end_index
        self.complement = complement
        self.is_canonical = is_canonical

    def get_subsequence(self, nucleotides: str) -> SeqType:
        # Note: The "complement" feature is not used; genbank annotation gives the resulting protein in terms
        # of the translation (on either forward or reverse strand), but we are modeling shotgun reads
        # (nucleotide substrings), not expression/protein levels.
        seq = nucleotides_to_z4(nucleotides[self.start_index:self.end_index])
        if self.complement:
            return reverse_complement_seq(seq)
        else:
            return seq

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{name}[{locus}:{start}:{end}]".format(
            name=self.name,
            locus=self.id,
            start=self.start_index,
            end=self.end_index
        )


def save_marker_to_disk(marker: Marker, filepath: Path):
    SeqIO.write([
        marker.to_seqrecord(description="")
    ], filepath, "fasta")


def get_marker_filepath(config, strain_accession: str, marker_id: str) -> Path:
    return (
            Path(config.database_cfg.data_dir)
            / "{acc}-{marker}.fasta".format(acc=strain_accession, marker=marker_id)
    )


class FastaLoader:
    def __init__(self, strain_accession: str, strain_fasta_path: Path, marker_entries: List[MarkerEntry]):
        self.strain_accession = strain_accession
        self.strain_fasta_path = strain_fasta_path
        self.marker_entries = marker_entries

    def parse_markers(self) -> Iterator[Marker]:
        for entry in self.marker_entries:
            if not isinstance(entry, SubseqMarkerEntry):
                raise StrainDatabaseParseError("Can't parse entry class {} in fasta strain record.".format(
                    entry.__class__.__name__
                ))

            marker_filepath = get_marker_filepath(cfg, self.strain_accession, entry.id)

            if marker_filepath.exists:
                marker_seq = nucleotides_to_z4(
                    str(SeqIO.read(marker_filepath, "fasta").seq)
                )
                marker = Marker(
                    name=entry.name,
                    id=entry.id,
                    seq=marker_seq,
                    canonical=entry.is_canonical,
                    metadata=MarkerMetadata(
                        parent_accession=self.strain_accession,
                        file_path=marker_filepath
                    )
                )
            else:
                seq = SeqIO.read(self.strain_fasta_path, "fasta").seq
                marker_seq = nucleotides_to_z4(str(seq[entry.start_pos - 1:entry.end_pos]))
                if not entry.is_negative_strand:
                    marker_seq = reverse_complement_seq(marker_seq)
                marker = Marker(
                    name=entry.name,
                    id=entry.id,
                    seq=marker_seq,
                    canonical=entry.is_canonical,
                    metadata=MarkerMetadata(
                        parent_accession=self.strain_accession,
                        file_path=marker_filepath
                    )
                )
                save_marker_to_disk(marker, marker_filepath)
            yield marker


class GenbankLoader:
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

        # TODO implement subseq entries!

        for entry in marker_entries:
            if isinstance(entry, TagMarkerEntry):
                self.tag_entries.append(entry)
            elif isinstance(entry, PrimerMarkerEntry):
                self.primer_entries.append(entry)
            else:
                raise NotImplementedError(
                    "Parsing entry class `{}` not implemented for genbank strains.".format(entry.__class__.__name__)
                )

        self.full_genome = None  # Lazy loading in get_full_genome()
        self.marker_max_len = int(marker_max_len)

    def get_full_genome(self, trim_debug=None) -> str:
        if self.full_genome is None:
            record = next(SeqIO.parse(self.genbank_filename, "genbank"))
            self.full_genome = str(record.seq)
            if trim_debug is not None:
                self.full_genome = self.full_genome[:trim_debug]
        return self.full_genome

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
            marker_filepath = get_marker_filepath(cfg, self.strain_accession, subseq_obj.id)
            marker = Marker(
                name=subseq_obj.name,
                id=subseq_obj.id,
                seq=subseq_obj.get_subsequence(self.get_full_genome()),
                canonical=subseq_obj.is_canonical,
                metadata=MarkerMetadata(
                    parent_accession=self.strain_accession,
                    file_path=marker_filepath
                )
            )
            save_marker_to_disk(marker, marker_filepath)
            yield marker

    def load_entries_from_disk(self) -> Iterator[Marker]:
        tag_entries_missed = []
        for tag_entry in self.tag_entries:
            marker_name = tag_entry.name
            marker_id = tag_entry.locus_tag
            marker_filepath = get_marker_filepath(cfg, self.strain_accession, marker_id)
            is_canonical = tag_entry.is_canonical
            try:
                yield self.load_marker_from_disk(marker_filepath, marker_name, marker_id, is_canonical)
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
            marker_filepath = get_marker_filepath(cfg, self.strain_accession, marker_id)
            is_canonical = primer_entry.is_canonical
            try:
                yield self.load_marker_from_disk(marker_filepath, marker_name, marker_id, is_canonical)
            except FileNotFoundError:
                primer_entries_missed.append(primer_entry)
            except StrainDatabaseParseError as e:
                logger.warning(str(e))
                primer_entries_missed.append(primer_entry)
        self.primer_entries = primer_entries_missed

    def load_marker_from_disk(self, filepath: Path, expected_marker_name: str, expected_marker_id: str, is_canonical: bool) -> Marker:
        record = next(SeqIO.parse(filepath, "fasta"))
        accession_token, name_token, id_token = record.id.strip().split("|")
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
                canonical=is_canonical,
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
                            complement=loc.strand == -1,
                            is_canonical=tag_entry.is_canonical
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
                    complement=False,
                    is_canonical=entry.is_canonical
                ))

        return subsequences

    def _regex_match_primers(self, forward: str, reverse: str) -> Union[None, Tuple[int, int]]:
        forward_primer_regex = self.parse_fasta_regex(forward)
        reverse_primer_regex = z4_to_nucleotides(
            reverse_complement_seq(
                nucleotides_to_z4(
                    self.parse_fasta_regex(reverse)
                )
            )
        )
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
    def __init__(self,
                 entries_file: Path,
                 marker_max_len: int,
                 force_refresh: bool = False,
                 load_full_genomes: bool = False):
        self.entries_file = entries_file
        self.marker_max_len = marker_max_len
        self.force_refresh = force_refresh
        self.load_full_genomes = load_full_genomes

    def strains(self) -> Iterator[Strain]:
        logger.info("Loading from JSON marker database file {}.".format(self.entries_file))
        logger.debug("Data will be saved to/load from: {}".format(cfg.database_cfg.data_dir))
        for strain_entry in self.strain_entries():
            if strain_entry.source == "genbank":
                genbank_path = fetch_genbank(strain_entry.accession,
                                             base_dir=cfg.database_cfg.data_dir,
                                             force_download=self.force_refresh)

                sequence_loader = GenbankLoader(
                    strain_accession=strain_entry.accession,
                    genbank_filename=genbank_path,
                    marker_entries=strain_entry.marker_entries,
                    marker_max_len=self.marker_max_len
                )

                strain_markers = list(sequence_loader.parse_markers(force_refresh=self.force_refresh))
                yield Strain(
                    id=strain_entry.accession,
                    markers=strain_markers,
                    metadata=StrainMetadata(
                        ncbi_accession=strain_entry.accession,
                        genus=strain_entry.genus,
                        species=strain_entry.species,
                        source_path=genbank_path
                    )
                )
            elif strain_entry.source == "fasta":
                fasta_path = fetch_fasta(strain_entry.accession,
                                         base_dir=cfg.database_cfg.data_dir,
                                         force_download=self.force_refresh)

                loader = FastaLoader(
                    strain_accession=strain_entry.accession,
                    strain_fasta_path=fasta_path,
                    marker_entries=strain_entry.marker_entries
                )

                strain_markers = list(loader.parse_markers())
                yield Strain(
                    id=strain_entry.accession,
                    markers=strain_markers,
                    metadata=StrainMetadata(
                        ncbi_accession=strain_entry.accession,
                        genus=strain_entry.genus,
                        species=strain_entry.species,
                        source_path=fasta_path
                    )
                )
            else:
                raise StrainDatabaseParseError("Unsupported strain entry source `{}`".format(strain_entry.source))

            if len(strain_markers) == 0:
                logger.warning("No markers parsed for entry {}.".format(strain_entry))
            else:
                logger.debug("Strain {} loaded with {} markers.".format(
                    strain_entry.accession,
                    len(strain_markers)
                ))

    def strain_entries(self) -> Iterator[StrainEntry]:
        """
        Deserialize JSON into StrainEntry instances.
        """
        with open(self.entries_file, "r") as f:
            for idx, strain_dict in enumerate(json.load(f)):
                yield StrainEntry.deserialize(strain_dict, idx)
