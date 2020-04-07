from abc import abstractmethod, ABCMeta
from scripts.fetch_genomes import fetch_sequences

import os
import re
from typing import List, Dict

_DEFAULT_DATA_DIR = "data"


class AbstractDatabase(metaclass=ABCMeta):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_markers(self, strain_id : str):
        pass


class SimpleCSVDatabase(AbstractDatabase):
    """
    A Simple implementation that treats each complete strain genome as a marker.
    """

    def __init__(self, csv_refs):
        """
        :param csv_refs: CSV file specifying accession numbers.
        """
        self.database = {}
        self.csv_refs = csv_refs

    def load(self) -> Dict[str, List[str]]:
        strain_info_map = fetch_sequences(refs_file_csv=self.csv_refs)
        self.database = {}

        for strain_accession in strain_info_map.keys():

            input_file_path = os.path.join(_DEFAULT_DATA_DIR, strain_accession + ".fasta")
            with open(input_file_path) as file:
                for i, line in enumerate(file):
                    genome = re.sub('[^AGCT]+', '', line.split(sep=" ")[-1])

            marker_sequences = [genome[:]]

            self.database[strain_accession] = marker_sequences
        return self.database

    def get_markers(self, strain_id: str) -> List[str]:
        return self.database[strain_id]
