from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Union

from chronostrain.config import cfg
from chronostrain.database.base import AbstractStrainDatabase, StrainEntryError, StrainNotFoundError
from chronostrain.model.bacteria import Marker, MarkerMetadata, Strain
from chronostrain.util.io.ncbi import fetch_fasta, fetch_genbank
from chronostrain.util.io.logger import logger


_complement_translation = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


# =====================================================================
# JSON entry dataclasses. Each class implements a deserialize() method.
# =====================================================================

@dataclass
class StrainEntry:
    name: str
    accession: str
    marker_entries: List[MarkerEntry]
    index: int

    def __str__(self):
        return "(Strain Entry #{}: {})".format(
            self.index,
            self.name
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
            name = json_dict["name"]
        except KeyError:
            raise StrainEntryError("Missing entry `name` from json entry.")

        try:
            accession = json_dict["accession"]
        except KeyError:
            raise StrainEntryError("Missing entry `accession` from json entry.")

        try:
            markers_arr = (json_dict["markers"])
        except KeyError:
            raise StrainEntryError("Missing entry `markers` from json entry.")

        marker_entries = []
        entry = StrainEntry(name=name, accession=accession, marker_entries=marker_entries, index=idx)
        for idx, marker_dict in enumerate(markers_arr):
            marker_entries.append(MarkerEntry.deserialize(marker_dict, idx, entry))
        return entry


@dataclass
class MarkerEntry:
    name: str
    index: int
    parent: StrainEntry

    @staticmethod
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> MarkerEntry:
        marker_type = entry_dict['type']
        if marker_type == 'tag':
            return TagMarkerEntry.deserialize(entry_dict, idx, parent)
        elif marker_type == 'primer':
            return PrimerMarkerEntry.deserialize(entry_dict, idx, parent)
        else:
            raise StrainEntryError("Unexpected type `{}` in marker entry {} of {}".format(marker_type, idx, parent))


@dataclass
class TagMarkerEntry(MarkerEntry):
    locus_id: str

    def __str__(self):
        return "(Tag Marker Entry #{} of {}: {})".format(
            self.index,
            self.parent.accession,
            self.name
        )

    def __repr__(self):
        return "{}(parent={},idx={},locus_id={})".format(
            self.__class__.__name__,
            self.parent.accession,
            self.index,
            self.locus_id
        )

    @staticmethod
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> TagMarkerEntry:
        return TagMarkerEntry(name=entry_dict['name'],
                              index=idx,
                              locus_id=entry_dict['locus_id'],
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

    @staticmethod
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> PrimerMarkerEntry:
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

    def get_subsequence(self, nucleotides: str):
        if self.complement:
            subsequence = nucleotides[self.start_index:self.end_index]
        else:
            subsequence = ''.join([
                _complement_translation[base] for base in nucleotides[self.start_index:self.end_index]
            ])
        return subsequence

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
                 fasta_filename: str,
                 genbank_filename: str,
                 marker_entries: List[MarkerEntry],
                 marker_max_len: int):
        self.fasta_filename = fasta_filename
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

        self.full_genome = None  # Lazy loading in get_full_genome() and get_genome_length()
        self.genome_length = None
        self.marker_max_len = int(marker_max_len)

    def get_full_genome(self, trim_debug=None) -> str:
        if self.full_genome is None:
            with open(self.fasta_filename) as file:
                lines = [re.sub('[^AGCT]+', '', line.split(sep=" ")[-1]) for line in file]
            self.full_genome = ''.join(lines)
            if trim_debug is not None:
                self.full_genome = self.full_genome[:trim_debug]
        return self.full_genome

    def get_genome_length(self):
        if self.genome_length is None:
            self.genome_length = len(self.get_full_genome())
        return self.genome_length

    def get_marker_subsequences(self) -> List[NucleotideSubsequence]:
        """
        Markers are expected to be a list of JSON objects of one of the following formats:
        (1) {'type': 'tag', 'name': <COMMON_NAME>, 'id': <NCBI_ID>}
        (2) {'type': 'primer', 'name': <COMMON_NAME>, 'forward': <FORWARD_SEQ>, 'reverse': <REV_SEQ>}

        This method parses either type, depending on the 'type' field.
        :return: A list of marker instances.
        """
        return self.get_subsequences_from_tags() + self.get_subsequences_from_primers()

    def get_subsequences_from_tags(self) -> List[NucleotideSubsequence]:
        if len(self.tag_entries) == 0:
            return []

        tags_to_entries = {
            entry.locus_id: entry
            for entry in self.tag_entries
        }

        tags_found = set()

        chunk_designation = None
        index_tag = ''
        subsequences = []

        with open(self.genbank_filename, 'rb') as genbank_file:
            logger.debug("Parsing tags {} from genbank file {}.".format(
                str(list(tags_to_entries.keys())),
                self.genbank_filename
            ))
            for line in genbank_file:
                # Split on >1 space
                split_line = re.split(r'\s{2,}', line.decode('utf-8').strip())

                if len(split_line) == 2:
                    chunk_designation = split_line[0]
                    if chunk_designation == 'gene':
                        # The first line of a gene chunk is the index tag
                        index_tag = split_line[1]

                if chunk_designation == 'gene':
                    if 'locus_tag' in split_line[-1]:
                        # Tags are declared by: /locus_tag=""
                        tag = split_line[-1].split('"')[1]
                        if tag in tags_to_entries:
                            tags_found.add(tag)
                            subsequences.append(self._parse_index_tag(index_tag, tags_to_entries[tag]))

        for tag, entry in tags_to_entries.items():
            if tag not in tags_found:
                logger.warn("Unable to find matches for tag entry {}.".format(entry))

        return subsequences

    def _parse_index_tag(self, index_tag: str, entry: TagMarkerEntry) -> NucleotideSubsequence:
        """
        Parses a genome location tag, e.g. "complement(312..432)" to create SubsequenceMetadata
        and pads indices by 100bp
        """
        indices = re.findall(r'[0-9]+', index_tag)
        if not len(indices) == 2:
            raise StrainEntryError('Encountered match to malformed tag: {}'.format(index_tag))

        max_index = self.get_genome_length()
        return NucleotideSubsequence(
            name=entry.name,
            id=entry.locus_id,
            start_index=max(0, int(indices[0]) - 100),
            end_index=min(int(indices[1]) + 100, max_index),
            complement='complement' in index_tag
        )

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
                logger.warn("Couldn't find matches for primer entry {}.".format(
                    str(entry)
                ))
            else:
                logger.debug("Found primer match: ({},{})".format(result[0], result[1]))
                subsequences.append(NucleotideSubsequence(
                    name=entry.name,
                    id='-'.join([entry.forward, entry.reverse]),
                    start_index=result[0],
                    end_index=result[1],
                    complement=False
                ))

        return subsequences

    def _regex_match_primers(self, forward: str, reverse: str) -> Union[None, Tuple[int, int]]:
        forward_primer_regex = self.parse_fasta_regex(forward)
        reverse_primer_regex = self.complement_regex(self.parse_fasta_regex(reverse[::-1]))
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

    @staticmethod
    def complement_regex(regex):
        complemented_regex = ''
        for char in regex:
            if char in _complement_translation.keys():
                complemented_regex += _complement_translation[char]
            else:
                complemented_regex += char
        return complemented_regex

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


# ====================================
# The main database implementation.
# ====================================

class JSONStrainDatabase(AbstractStrainDatabase):
    """
    A implementation which defines strains as collections of markers, using the JSON format found in the example.
    """

    def __init__(self, entries_file, marker_max_len):
        """
        :param entries_file: JSON filename specifying accession numbers and marker locus tags.
        """
        self.strains = dict()  # accession -> Strain
        self.entries_file = entries_file
        self.marker_max_len = marker_max_len
        super().__init__()

    def __load__(self):
        logger.info("Loading from JSON marker database file {}.".format(self.entries_file))
        for strain_entry in self.strain_entries():
            fasta_filename = fetch_fasta(strain_entry.accession, base_dir=cfg.database_cfg.data_dir)
            genbank_filename = fetch_genbank(strain_entry.accession, base_dir=cfg.database_cfg.data_dir)

            # TODO only do regex searches if can't load from disk. Try to load from disk first
            # TODO when implementing this, be wary of copy numbers (need to decide when/where to handle it.)

            sequence_loader = SubsequenceLoader(
                fasta_filename=fasta_filename,
                genbank_filename=genbank_filename,
                marker_entries=strain_entry.marker_entries,
                marker_max_len=self.marker_max_len
            )

            genome = sequence_loader.get_full_genome()
            markers = []
            for subsequence_data in sequence_loader.get_marker_subsequences():
                markers.append(Marker(
                    name=subsequence_data.id,
                    seq=subsequence_data.get_subsequence(genome),
                    metadata=MarkerMetadata(
                        strain_accession=strain_entry.accession,
                        subseq_name=subsequence_data.name
                    )
                ))
            self.strains[strain_entry.accession] = Strain(
                name="{}:{}".format(strain_entry.name, strain_entry.accession),
                markers=markers,
                genome_length=sequence_loader.get_genome_length()
            )
            if len(markers) == 0:
                logger.warn("No markers parsed for entry {}.".format(strain_entry))

    def get_strain(self, strain_id: str) -> Strain:
        if strain_id not in self.strains:
            raise StrainNotFoundError(strain_id)

        return self.strains[strain_id]

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def strain_entries(self) -> List[StrainEntry]:
        """
        Deserialize JSON into StrainEntry instances.
        """
        with open(self.entries_file, "r") as f:
            return [StrainEntry.deserialize(strain_dict, idx) for idx, strain_dict in enumerate(json.load(f))]

    @staticmethod
    def marker_filename(accession: str, name: str):
        return "{acc}-{seq}.fasta".format(acc=accession, seq=name)

    def dump_markers_to_fasta(self):
        """
        For each accession-marker pair, write the sequence to disk.
        Resulting filename is `<accession>-<marker name>.fasta`.
        :return:
        """
        resulting_filenames = []

        for accession in self.strains.keys():
            for marker in self.strains[accession].markers:
                filepath = os.path.join(
                    cfg.database_cfg.data_dir,
                    self.marker_filename(accession, marker.metadata.subseq_name)
                )

                resulting_filenames.append(filepath)

                with open(filepath, 'w') as f:
                    f.write('>' + accession + '-' + marker.metadata.subseq_name + '\n')
                    for i in range(len(marker.seq)):
                        f.write(marker.seq[i])
                        if (i + 1) % 70 == 0:
                            f.write('\n')

        return resulting_filenames
