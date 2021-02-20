import csv
import os
import re
from typing import List

from database.base import AbstractStrainDatabase, StrainEntryError, _DEFAULT_DATA_DIR
from model.bacteria import Strain, Marker
from util.io.fetch_genomes import fetch_filenames
from util.io.logger import logger


class SimpleCSVStrainDatabase(AbstractStrainDatabase):
    """
    A Simple implementation that treats each complete strain genome as a marker.
    """

    def __init__(self, csv_refs, trim_debug=None):
        """
        :param csv_refs: CSV file specifying accession numbers.
        """
        if trim_debug:
            logger.debug("[SimpleCSVStrainDatabase: initialized in debug mode. Trim length = {L}]".format(L=trim_debug))
        self.strains = dict()
        self.csv_refs = csv_refs
        self.trim_debug = trim_debug
        super().__init__()

    def __load__(self):
        for strain_name, accession, filename in self.strain_entries():
            input_file_path = os.path.join(_DEFAULT_DATA_DIR, accession + ".fasta")
            with open(input_file_path) as file:
                lines = [re.sub('[^AGCT]+', '', line.split(sep=" ")[-1]) for line in file]
            genome = ''.join(lines)
            if self.trim_debug is not None:
                genome = genome[:self.trim_debug]
            markers = [Marker(name=strain_name, seq=genome)]  # Each genome's marker is its own genome.
            self.strains[accession] = Strain(name="{}:{}".format(strain_name, accession), markers=markers)

    def get_strain(self, strain_id: str) -> Strain:
        """
        :param strain_id: An accession string (NCBI format).
        :return: A Strain instance.
        """
        return self.strains[strain_id]

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def strain_entries(self):
        """
        Read CSV file, and download FASTA from accessions if doesn't exist.
        :return: a dictionary mapping accessions to strain-accession-filename wrappers.
        """
        line_count = 0
        with open(self.csv_refs, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                line_count += 1
                if line_count == 1:
                    continue
                if len(row) != 2:
                    raise StrainEntryError("{} -- Expected 2 tokens on line {}, got {}.".format(
                        self.csv_refs, line_count, len(row))
                    )
                strain_name = row[0]
                accession = row[1]
                filename = fetch_filenames(accession)[0]
                yield strain_name, accession, filename
        logger.info("Found {} records.".format(line_count - 1))
