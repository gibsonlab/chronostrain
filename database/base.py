from abc import abstractmethod, ABCMeta

from model.bacteria import Strain, Marker, MarkerMetadata
from util.io.fetch_genomes import fetch_sequences, SubsequenceMetadata

import os
import re
from typing import List
from util.io.logger import logger

_DEFAULT_DATA_DIR = "data"


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
        strain_info_map = fetch_sequences(refs_file_json=self.json_refs)
        for strain_accession in strain_info_map.keys():

            input_file_path = os.path.join(_DEFAULT_DATA_DIR, strain_accession + ".fasta")
            with open(input_file_path) as file:
                lines = [re.sub('[^AGCT]+', '', line.split(sep=" ")[-1]) for line in file]
            genome = ''.join(lines)
            if self.trim_debug is not None:
                genome = genome[:self.trim_debug]
            markers = []
            if 'subsequences' in strain_info_map[strain_accession].keys():
                for subsequence in strain_info_map[strain_accession]['subsequences'].keys():
                    markers.append(Marker(
                        name = subsequence,
                        seq = strain_info_map[strain_accession]['subsequences'][subsequence].get_subsequence(genome),
                        metadata = MarkerMetadata(parent=strain_accession,
                            subsequence_name = strain_info_map[strain_accession]['subsequences'][subsequence].name,
                            parent_genome_length = len(genome)
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
