from abc import abstractmethod, ABCMeta
from scripts.fetch_genomes import fetch_sequences

import os
import re
from typing import List, Dict

_data_dir = "data"

class AbstractDatabase(metaclass=ABCMeta):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get_markers(self, strain_id : str):
        pass


class SimpleCSVDatabase(AbstractDatabase):

    def __init__(self, src):
        self.database = {}
        self.src = src

    def load(self) -> Dict[str, List[str]]:

        strain_info_map = fetch_sequences(refs_file_csv=self.src)
        self.database = {}

        for strain_accession in strain_info_map.keys():

            input_file_path = os.path.join(_data_dir, strain_accession + ".fasta")
            with open(input_file_path) as file:
                for i, line in enumerate(file):
                    genome = re.sub('[^AGCT]+', '', line.split(sep=" ")[-1])

            # sequences = [SeqRecord(Seq(genome[0:50])),
            #              SeqRecord(Seq(genome[110:160])),
            #              SeqRecord(Seq(genome[170:220]))]
            marker_sequences = [genome[:]]

            self.database[strain_accession] = marker_sequences
        return self.database

    def get_markers(self, strain_id : str) -> List[str]:
        return self.database[strain_id]
