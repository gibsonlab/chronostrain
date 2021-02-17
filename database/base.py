from abc import abstractmethod, ABCMeta

from model.bacteria import Strain, Marker, MarkerMetadata
from util.io.fetch_genomes import fetch_filenames

import os
import re
import json
from typing import List
from util.io.logger import logger
from dataclasses import dataclass

_DEFAULT_DATA_DIR = "data"
_MAXIMUM_MARKER_LEN = 2000


class AbstractStrainDatabase(metaclass=ABCMeta):
    def __init__(self):
        self.__load__()

    @abstractmethod
    def __load__(self):
        pass

    @abstractmethod
    def get_strain(self, strain_id: str) -> Strain:
        pass

    @abstractmethod
    def all_strains(self) -> List[Strain]:
        pass

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return [self.get_strain(s_id) for s_id in strain_ids]


class SimpleCSVStrainDatabase(AbstractStrainDatabase):
    """
    A Simple implementation that treats each complete strain genome and optional specified subsequences as markers.
    """

    def __init__(self, json_refs, trim_debug=None):
        """
        :param json_refs: JSON file specifying accession numbers and marker locus tags.
        """
        if trim_debug:
            logger.debug("[SimpleCSVStrainDatabase: initialized in debug mode. Trim length = {L}]".format(L=trim_debug))
        self.strain_to_markers = {}
        self.json_refs = json_refs
        self.trim_debug = trim_debug
        super().__init__()

    def __load__(self):
        strain_info_map = self.fetch_sequences()
        for strain_accession in strain_info_map.keys():

            sequence_loader = strain_info_map[strain_accession]['sequenceLoader']
            genome = sequence_loader.get_full_genome(self.trim_debug)
            markers = []
            for subsequence_data in sequence_loader.get_marker_subsequences():
                markers.append(Marker(
                    name = subsequence_data.ID,
                    seq = subsequence_data.get_subsequence(genome),
                    metadata = MarkerMetadata(parent=strain_accession,
                        subsequence_name = subsequence_data.name,
                        parent_genome_length = sequence_loader.get_genome_length()
                    )
                ))
            if len(markers) > 0:
                self.strain_to_markers[strain_accession] = markers

    def get_strain(self, strain_id: str) -> Strain:
        return Strain(
            name=strain_id,
            markers=self.strain_to_markers[strain_id],
            genome_length = self.strain_to_markers[strain_id][0].metadata.parent_genome_length
        )

    def all_strains(self) -> List[Strain]:
        return [
            Strain(name=s_id, markers=markers, genome_length=markers[0].metadata.parent_genome_length)
            for (s_id, markers) in self.strain_to_markers.items()
        ]


    def fetch_sequences(self):
        """
        Read JSON file, and download FASTA from accessions if doesn't exist.
        :return: a dictionary mapping accessions to strain-accession-filename-subsequences
                 wrappers.
        """
        strains_map = {}

        with open(self.json_refs, "rb") as f:
            strain_infos = json.load(f, object_hook = lambda info: StrainInfo(info))
            for info in strain_infos:
                info.validate()
                fasta_filename, genbank_filename = fetch_filenames(info['accession'])
                strains_map[info['accession']] = {
                    "strain": info['name'],
                    "accession": info['accession'],
                    "file": fasta_filename,
                    "sequenceLoader": SubsequenceLoader(fasta_filename, genbank_filename, info['markers'])
                }

        logger.info("Found {} records.".format(len(strains_map.keys())))
        return strains_map
    

class SubsequenceLoader:


    def __init__(self, fasta_filename, genbank_filename, marker_info: list):
        self.fasta_filename = fasta_filename
        self.genbank_filename = genbank_filename
        self.marker_info = marker_info
        self.full_genome = None # Lazy loading in get_full_genome() and get_genome_length()
        self.genome_length = None


    def get_full_genome(self, trim_debug=None):
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


    def get_marker_subsequences(self):
        locus_tags_to_names = {}
        names_to_primers = {}
        for marker in self.marker_info:
            if marker['type'] == 'tag':
                locus_tags_to_names[marker['id']] = marker['name']
            elif marker['type'] == 'primer':
                names_to_primers[marker['name']] = (marker['forward'], marker['reverse'])
        return self.get_subsequences_from_locus_tags(locus_tags_to_names) + self.get_subsequences_from_primers(names_to_primers)


    def parse_index_tag(self, index_tag: str, sequence_name, locus_tag):
        '''
        Parses a genome location tag, e.g. "complement(312..432)" to create SubsequenceMetadata
        and pads indices by 100bp
        '''
        indices = re.findall(r'[0-9]+', index_tag)
        if not len(indices) == 2:
            logger.warning('Encountered match to malformed tag: ' + index_tag)
            return None
        
        max_index = self.get_genome_length()
        return SubsequenceMetadata(
            name=sequence_name, start_index=max(0, int(indices[0])-100), ID=locus_tag,
            end_index=min(int(indices[1])+100, max_index), complement='complement' in index_tag
        )
        

    def get_subsequences_from_locus_tags(self, tags_to_names: dict):
        '''
        :param filename: Name of local genbank file
        :param tags_to_names: Dictionary of locus tags to the name of their subsequence
        :return: List of SubsequenceMetadata deriving from all matched tags in the genbank file
        '''
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


    def parse_fasta_regex(self, sequence):
        fasta_translation = {
            'R':'[AG]',  'Y':'[CT]',   'K':'[GT]',
            'M':'[AC]',  'S':'[CG]',   'W':'[AT]',
            'B':'[CGT]', 'D':'[AGT]',  'H':'[ACT]',
            'V':'[ACG]', 'N':'[ACGT]', 'A':'A',
            'C':'C',     'G':'G',      'T':'T'
        }
        sequence_regex = ''
        for char in sequence:
            sequence_regex += fasta_translation[char]
        return sequence_regex


    def complement_regex(self, regex):
        complement_translation = {'A': 'T', 'T':'A', 'G':'C', 'C':'G'}
        complemented_regex = ''
        for char in regex:
            if char in complement_translation.keys():
                complemented_regex += complement_translation[char]
            else:
                complemented_regex += char
        return complemented_regex


    def find_primer_match(self, forward_regex, reverse_regex):
        best_match = (None, _MAXIMUM_MARKER_LEN)
        forward_matches = list(re.finditer(forward_regex, self.get_full_genome()))
        reverse_matches = list(re.finditer(reverse_regex, self.get_full_genome()))

        for forward_match in forward_matches:
            for reverse_match in reverse_matches:
                match_length = reverse_match.end()-forward_match.start()
                if match_length < best_match[1] and match_length > 0:
                    best_match = ((forward_match.start(), reverse_match.end()),match_length)
        return best_match[0]


    def get_subsequences_from_primers(self, names_to_primers: dict):
        subsequences = []

        for marker_name in names_to_primers.keys():
            forward_primer_regex = self.parse_fasta_regex(names_to_primers[marker_name][0])
            # Reverse is read from the opposite end of the complement strand, so we reverse and complement
            reverse_primer_regex = self.complement_regex(
                self.parse_fasta_regex(names_to_primers[marker_name][1][::-1])
            )
            match_indices = self.find_primer_match(forward_primer_regex, reverse_primer_regex)
            if match_indices is not None:
                subsequences.append(SubsequenceMetadata(
                    name = marker_name, ID = '-'.join(names_to_primers[marker_name]),
                    start_index = match_indices[0], end_index = match_indices[1],
                    complement = False
                ))
                continue
            forward_primer_regex = self.complement_regex(self.parse_fasta_regex(names_to_primers[marker_name][0][::-1]))
            reverse_primer_regex = self.parse_fasta_regex(names_to_primers[marker_name][1])
            match_indices = self.find_primer_match(reverse_primer_regex, forward_primer_regex)
            if match_indices is not None:
                subsequences.append(SubsequenceMetadata(
                    name = marker_name, ID = '-'.join(names_to_primers[marker_name]),
                    start_index = match_indices[0], end_index = match_indices[1],
                    complement = True
                ))

        return subsequences


class StrainInfo(dict):
    def __getattr__(self, key):
        return self[key]
 
    def __setattr__(self, key, value):
        if key not in ['name', 'accession', 'markers']:
            logger.error('Invalid attribute name in refs JSON')
            raise KeyError
        self[lower(key)] = value

    def validate(self):
        if 'name' not in self.keys() or 'accession' not in self.keys() or 'markers' not in self.keys():
            logger.error('Missing name, accession, or markers for entry in refs JSON: ' + str(self))
            class StrainInfoIncompleteError(Exception):
                pass
            raise StrainInfoIncompleteError


@dataclass
class SubsequenceMetadata:
    name: str
    ID: str
    start_index: int
    end_index: int
    complement: bool # Indicates whether the intended sequence is the complement of the given sequence
    
    def get_subsequence(self, genome: str):
        complement_translation = {'A': 'T', 'T':'A', 'G':'C', 'C':'G'}
        complement_func = (lambda base: complement_translation[base]) if self.complement else lambda base: base
        subsequence = ''
        for base in genome[self.start_index:self.end_index]:
            subsequence += complement_func(base)
        return subsequence