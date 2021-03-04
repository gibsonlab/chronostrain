import os
import re
import json
from typing import List, Tuple, Union

from chronostrain.config import cfg
from chronostrain.database.base import AbstractStrainDatabase, StrainEntryError, StrainNotFoundError
from chronostrain.model.bacteria import Marker, MarkerMetadata, Strain
from chronostrain.util.io.ncbi import fetch_fasta, fetch_genbank
from chronostrain.util.io.logger import logger


_complement_translation = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def parse_strain_info(json_dict) -> Tuple[str, str, List[dict]]:
    try:
        name = json_dict["name"]
    except KeyError:
        raise StrainEntryError("Missing entry `name` from json entry.")

    try:
        accession = json_dict["accession"]
    except KeyError:
        raise StrainEntryError("Missing entry `accession` from json entry.")

    try:
        markers = json_dict["markers"]
    except KeyError:
        raise StrainEntryError("Missing entry `markers` from json entry.")

    return name, accession, markers


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

    def __init__(self, fasta_filename, genbank_filename, marker_info: List[dict], marker_max_len: int):
        self.fasta_filename = fasta_filename
        self.genbank_filename = genbank_filename
        self.marker_info = marker_info
        self.full_genome = None  # Lazy loading in get_full_genome() and get_genome_length()
        self.genome_length = None
        self.marker_max_len = marker_max_len

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
        locus_tags_to_names = {}
        names_to_primers = {}
        for marker in self.marker_info:
            if marker['type'] == 'tag':
                locus_tags_to_names[marker['id']] = marker['name']
            elif marker['type'] == 'primer':
                names_to_primers[marker['name']] = (marker['forward'], marker['reverse'])
        return self.get_subsequences_from_locus_tags(locus_tags_to_names) + self.get_subsequences_from_primers(
            names_to_primers)

    def parse_index_tag(self, index_tag: str, sequence_name, locus_tag) -> NucleotideSubsequence:
        """
        Parses a genome location tag, e.g. "complement(312..432)" to create SubsequenceMetadata
        and pads indices by 100bp
        """
        indices = re.findall(r'[0-9]+', index_tag)
        if not len(indices) == 2:
            raise StrainEntryError('Encountered match to malformed tag: {}'.format(index_tag))

        max_index = self.get_genome_length()
        return NucleotideSubsequence(
            name=sequence_name,
            id=locus_tag,
            start_index=max(0, int(indices[0]) - 100),
            end_index=min(int(indices[1]) + 100, max_index),
            complement='complement' in index_tag
        )

    def get_subsequences_from_locus_tags(self, tags_to_names: dict) -> List[NucleotideSubsequence]:
        """
        :param tags_to_names: Dictionary of locus tags to the name of their subsequence
        :return: List of SubsequenceMetadata deriving from all matched tags in the genbank file
        """
        chunk_designation = None
        potential_index_tag = ''
        locus_tags = set(tags_to_names.keys())
        subsequences = []

        with open(self.genbank_filename, 'rb') as genbank_file:
            for line in genbank_file:
                # Split on >1 space
                split_line = re.split(r'\s{2,}', line.decode('utf-8').strip())

                if len(split_line) == 2:
                    chunk_designation = split_line[0]
                    if chunk_designation == 'gene':
                        # The first line of a gene chunk is the index tag
                        potential_index_tag = split_line[1]

                if chunk_designation == 'gene':
                    if 'locus_tag' in split_line[-1]:
                        # Tags are declared by: /locus_tag=""
                        tag = split_line[-1].split('"')[1]
                        if tag in locus_tags:
                            subsequences.append(self.parse_index_tag(potential_index_tag, tags_to_names[tag], tag))

        return [sequence for sequence in subsequences if sequence is not None]

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

    def get_subsequences_from_primers(self, names_to_primers: dict) -> List[NucleotideSubsequence]:
        subsequences = []

        for marker_name in names_to_primers.keys():
            forward_primer_regex = self.parse_fasta_regex(names_to_primers[marker_name][0])
            # Reverse is read from the opposite end of the complement strand, so we reverse and complement
            reverse_primer_regex = self.complement_regex(
                self.parse_fasta_regex(names_to_primers[marker_name][1][::-1])
            )
            match_indices = self.find_primer_match(forward_primer_regex, reverse_primer_regex)
            if match_indices is not None:
                subsequences.append(NucleotideSubsequence(
                    name=marker_name,
                    id='-'.join(names_to_primers[marker_name]),
                    start_index=match_indices[0],
                    end_index=match_indices[1],
                    complement=False
                ))
                continue
            forward_primer_regex = self.complement_regex(self.parse_fasta_regex(names_to_primers[marker_name][0][::-1]))
            reverse_primer_regex = self.parse_fasta_regex(names_to_primers[marker_name][1])
            match_indices = self.find_primer_match(reverse_primer_regex, forward_primer_regex)
            if match_indices is not None:
                subsequences.append(NucleotideSubsequence(
                    name=marker_name,
                    id='-'.join(names_to_primers[marker_name]),
                    start_index=match_indices[0],
                    end_index=match_indices[1],
                    complement=False
                ))

        return subsequences


class JSONStrainDatabase(AbstractStrainDatabase):
    """
    A Simple implementation that treats each complete strain genome and optional specified subsequences as markers.
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
        for strain_name, strain_accession, strain_markers in self.strain_entries():
            fasta_filename = fetch_fasta(strain_accession, base_dir=cfg.database_cfg.data_dir)
            genbank_filename = fetch_genbank(strain_accession, base_dir=cfg.database_cfg.data_dir)
            sequence_loader = SubsequenceLoader(
                fasta_filename=fasta_filename,
                genbank_filename=genbank_filename,
                marker_info=strain_markers,
                marker_max_len=self.marker_max_len
            )

            genome = sequence_loader.get_full_genome()
            markers = []
            for subsequence_data in sequence_loader.get_marker_subsequences():
                markers.append(Marker(
                    name=subsequence_data.id,
                    seq=subsequence_data.get_subsequence(genome),
                    metadata=MarkerMetadata(
                        strain_accession=strain_accession,
                        subseq_name=subsequence_data.name
                    )
                ))
            self.strains[strain_accession] = Strain(
                name="{}:{}".format(strain_name, strain_accession),
                markers=markers,
                genome_length=sequence_loader.get_genome_length()
            )
            if len(markers) == 0:
                logger.warn("No markers parsed for strain {}.".format(strain_accession))

    def get_strain(self, strain_id: str) -> Strain:
        if strain_id not in self.strains:
            raise StrainNotFoundError(strain_id)

        return self.strains[strain_id]

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def strain_entries(self) -> Tuple[str, str, dict]:
        """
        Read JSON file, and download FASTA from accessions if doesn't exist.
        :return: a dictionary mapping accessions to strain-accession-filename-subsequences
                 wrappers.
        """
        with open(self.entries_file, "r") as f:
            for strain_dict in json.load(f):
                yield parse_strain_info(strain_dict)

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
                    "{acc}-{seq}.fasta".format(acc=accession, seq=marker.metadata.subseq_name)
                )

                resulting_filenames.append(filepath)

                with open(filepath, 'w') as f:
                    f.write('>' + accession + '-' + marker.metadata.subseq_name + '\n')
                    for i in range(len(marker.seq)):
                        f.write(marker.seq[i])
                        if (i + 1) % 70 == 0:
                            f.write('\n')

        return resulting_filenames
