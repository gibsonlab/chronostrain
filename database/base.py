from abc import abstractmethod, ABCMeta

from model.bacteria import Strain, Marker
from scripts.fetch_genomes import fetch_sequences

import os
import re
from typing import List

_DEFAULT_DATA_DIR = "data"


class AbstractStrainDatabase(metaclass=ABCMeta):
    def __init__(self):
        self.load()

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_strain(self, strain_id: str) -> Strain:
        pass

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return [self.get_strain(s_id) for s_id in strain_ids]



class SimpleCSVStrainDatabase(AbstractStrainDatabase):
    """
    A Simple implementation that treats each complete strain genome as a marker.
    """

    def __init__(self, csv_refs):
        """
        :param csv_refs: CSV file specifying accession numbers.
        """
        self.strain_to_markers = {}
        self.csv_refs = csv_refs
        super().__init__()

    def __load__(self):
        strain_info_map = fetch_sequences(refs_file_csv=self.csv_refs)
        for strain_accession in strain_info_map.keys():

            input_file_path = os.path.join(_DEFAULT_DATA_DIR, strain_accession + ".fasta")
            with open(input_file_path) as file:
                for i, line in enumerate(file):
                    genome = re.sub('[^AGCT]+', '', line.split(sep=" ")[-1])

            markers = [Marker(name=strain_accession, seq=genome)]  # Each genome's marker is its own genome.
            self.strain_to_markers[strain_accession] = markers

    def get_markers(self, strain_id: str) -> Strain:
        return Strain(name=strain_id, markers=self.strain_to_markers[strain_id])
