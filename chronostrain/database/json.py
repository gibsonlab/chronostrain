from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Union

from Bio import SeqIO

from chronostrain.config import cfg
from chronostrain.database.base import AbstractStrainDatabase, StrainEntryError, StrainNotFoundError
from chronostrain.model.bacteria import Marker, MarkerMetadata, Strain, StrainMetadata
from chronostrain.util.ncbi import fetch_fasta, fetch_genbank
from . import logger

from chronostrain.util.sequences import complement_seq


# =====================================================================
# JSON entry dataclasses. Each class implements a deserialize() method.
# =====================================================================


@dataclass
class StrainEntry:
    genus: str
    species: str
    accession: str
    marker_entries: List[MarkerEntry]
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
            raise StrainEntryError("Missing entry `genus` from json entry.")

        try:
            species = json_dict["species"]
        except KeyError:
            raise StrainEntryError("Missing entry `species` from json entry.")

        try:
            accession = json_dict["accession"]
        except KeyError:
            raise StrainEntryError("Missing entry `accession` from json entry.")

        try:
            markers_arr = (json_dict["markers"])
        except KeyError:
            raise StrainEntryError("Missing entry `markers` from json entry.")

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
    def deserialize(entry_dict: dict, idx: int, parent: StrainEntry) -> TagMarkerEntry:
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
                logger.warn("Unable to find matches for tag entry {}.".format(entry))

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


# ====================================
# The main database implementation.
# ====================================

class JSONStrainDatabase(AbstractStrainDatabase):
    """
    A implementation which defines strains as collections of markers, using the JSON format found in the example.
    """

    def __init__(self, entries_file, marker_max_len, force_refresh: bool = False):
        """
        :param entries_file: JSON filename specifying accession numbers and marker locus tags.
        """
        self.strains = dict()  # accession -> Strain
        self.entries_file = entries_file
        self.marker_max_len = marker_max_len
        super().__init__(force_refresh=force_refresh)

    def __load__(self, force_refresh: bool = False):
        logger.info("Loading from JSON marker database file {}.".format(self.entries_file))
        logger.debug("Data will be saved to/load from: {}".format(cfg.database_cfg.data_dir))
        for strain_entry in self.strain_entries():
            fasta_filename = fetch_fasta(strain_entry.accession,
                                         base_dir=cfg.database_cfg.data_dir,
                                         force_download=force_refresh)
            genbank_filename = fetch_genbank(strain_entry.accession,
                                             base_dir=cfg.database_cfg.data_dir,
                                             force_download=force_refresh)

            # TODO only do regex searches if can't load from disk. Try to load from disk first
            #  when implementing this, be wary of copy numbers (need to decide when/where to handle it.)

            sequence_loader = SubsequenceLoader(
                fasta_filename=fasta_filename,
                genbank_filename=genbank_filename,
                marker_entries=strain_entry.marker_entries,
                marker_max_len=self.marker_max_len
            )

            genome = sequence_loader.get_full_genome()
            markers = []
            for subsequence_data in sequence_loader.get_marker_subsequences():
                marker_filepath = os.path.join(
                    cfg.database_cfg.data_dir,
                    self.marker_filename(strain_entry.accession, subsequence_data.name)
                )
                markers.append(Marker(
                    name=subsequence_data.name,
                    seq=subsequence_data.get_subsequence(genome),
                    metadata=MarkerMetadata(
                        gene_id=subsequence_data.id,
                        file_path=marker_filepath
                    )
                ))
                self.save_marker_to_fasta(strain_entry.accession, markers[-1], marker_filepath)

            self.strains[strain_entry.accession] = Strain(
                id=strain_entry.accession,
                markers=markers,
                genome_length=sequence_loader.get_genome_length(),
                metadata=StrainMetadata(
                    ncbi_accession=strain_entry.accession,
                    genus=strain_entry.genus,
                    species=strain_entry.species,
                    file_path=fasta_filename
                ))

            if len(markers) == 0:
                logger.warn("No markers parsed for entry {}.".format(strain_entry))
            else:
                logger.debug("Strain {} loaded with {} markers.".format(
                    strain_entry.accession,
                    len(markers)
                ))

        # Save multi-fasta.
        self.multifasta_file = os.path.join(cfg.database_cfg.data_dir, 'marker_multifasta.fa')
        self.save_markers_to_multifasta(filepath=self.multifasta_file, force_refresh=force_refresh)
        logger.debug("Multi-fasta file: {}".format(self.multifasta_file))

    def get_strain(self, strain_id: str) -> Strain:
        if strain_id not in self.strains:
            raise StrainNotFoundError(strain_id)

        return self.strains[strain_id]

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def num_strains(self) -> int:
        return len(self.strains)

    def strain_entries(self) -> List[StrainEntry]:
        """
        Deserialize JSON into StrainEntry instances.
        """
        with open(self.entries_file, "r") as f:
            return [StrainEntry.deserialize(strain_dict, idx) for idx, strain_dict in enumerate(json.load(f))]

    def get_marker_filenames(self):
        filenames = []
        for strain in self.all_strains():
            for marker in strain.markers:
                filenames.append(marker.metadata.file_path)
        return filenames

    def get_multifasta_file(self):
        return self.multifasta_file

    def strain_markers_to_fasta(self, strain_id: str, out_path: str):
        strain = self.get_strain(strain_id)
        with open(out_path, "w") as marker_file:
            for marker in strain.markers:
                print(">{}|{}|{}".format(strain_id, marker.name, marker.metadata.gene_id), file=marker_file)
                print(marker.seq, file=marker_file)

    def save_markers_to_multifasta(self,
                                   filepath: str,
                                   force_refresh: bool = True):
        if not force_refresh and os.path.exists(filepath):
            logger.debug("Multi-fasta file already exists; skipping creation.".format(
                filepath
            ))
        else:
            marker_files = self.get_marker_filenames()
            with open(filepath, 'w') as multifa:
                for filename in marker_files:
                    with open(filename, 'r') as marker_file:
                        for line in marker_file:
                            multifa.write(line)
                    multifa.write('\n')

    @staticmethod
    def marker_filename(accession: str, name: str):
        return "{acc}-{seq}.fasta".format(acc=accession, seq=name)

    @staticmethod
    def save_marker_to_fasta(strain_id: str, marker: Marker, filepath):
        with open(filepath, 'w') as f:
            print(">{}|{}|{}".format(
                strain_id,
                marker.name,
                marker.metadata.gene_id
            ), file=f)
            for i in range(len(marker.seq)):
                f.write(marker.seq[i])
                if (i + 1) % 70 == 0:
                    f.write('\n')
