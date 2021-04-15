import csv
import re
from pathlib import Path
from typing import List, Optional

from chronostrain.config import cfg
from chronostrain.database.base import AbstractStrainDatabase, StrainEntryError, StrainNotFoundError
from chronostrain.model.bacteria import Strain, Marker, StrainMetadata, MarkerMetadata
from chronostrain.util.ncbi import fetch_fasta
from . import logger


class SimpleCSVStrainDatabase(AbstractStrainDatabase):
    """
    A Simple implementation that treats each complete strain genome as a marker.
    """

    def __init__(self, entries_file, trim_debug: Optional[int] = None, force_refresh: bool = False):
        """
        :param entries_file: CSV file specifying accession numbers.
        :param trim_debug: If an int is passed, the genome is trimmed down to the first `trim_debug` characters.
        If None, entire genome is used.
        """
        self.strains = dict()
        self.entries_file = entries_file
        if trim_debug is not None:
            logger.debug("[SimpleCSVStrainDatabase: initialized in debug mode. Trim length = {L}]".format(L=trim_debug))
            self.trim_debug = int(trim_debug)
        super().__init__(force_refresh=force_refresh)

    def __load__(self, force_refresh: bool = False):
        logger.info("Loading from CSV marker database file {}.".format(self.entries_file))
        for strain_name, accession, fasta_filename in self._strain_entries(force_refresh):
            with open(fasta_filename, "r") as file:
                lines = [re.sub('[^AGCT]+', '', line.split(sep=" ")[-1]) for line in file]
            genome = ''.join(lines)
            if self.trim_debug is not None:
                markers = [
                    Marker(
                        name="Genome_{}[{}:{}]".format(accession, 0, self.trim_debug),
                        seq=genome[:self.trim_debug],
                        metadata=MarkerMetadata(gene_id="GENOME", file_path=fasta_filename)
                    )
                ]
            else:
                markers = [
                    Marker(
                        name="Genome_{}".format(accession),
                        seq=genome,
                        metadata=MarkerMetadata(gene_id="GENOME", file_path=fasta_filename)
                    )
                ]  # Each genome's marker is its own genome.
            self.strains[accession] = Strain(
                id=accession,
                markers=markers,
                genome_length=len(genome),
                metadata=StrainMetadata(
                    ncbi_accession=accession,
                    genus="",  # TODO: make it up-to-date with JSONDatabase.
                    species=strain_name,
                    file_path=fasta_filename
                )
            )

    def get_strain(self, strain_id: str) -> Strain:
        """
        :param strain_id: An accession string (NCBI format).
        :return: A Strain instance.
        """
        if strain_id not in self.strains:
            raise StrainNotFoundError(strain_id)

        return self.strains[strain_id]

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def num_strains(self) -> int:
        return len(self.strains)

    def strain_markers_to_fasta(self, strain_id: str, out_path: Path, file_mode: str = "w"):
        if self.trim_debug is None:
            raise RuntimeError(
                "Strains loaded by {} uses entire genomes as markers. Avoid calling this implementation!".format(
                    self.__class__.__name__
                )
            )
        else:
            super().strain_markers_to_fasta(strain_id, out_path, file_mode=file_mode)

    def _strain_entries(self, force_refresh: bool):
        """
        Read CSV file, and download FASTA from accessions if doesn't exist.
        :return: a dictionary mapping accessions to strain-accession-filename wrappers.
        """
        line_count = 0
        with open(self.entries_file, "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                line_count += 1
                if line_count == 1:
                    continue
                if len(row) != 2:
                    raise StrainEntryError("{} -- Expected 2 tokens on line {}, got {}.".format(
                        self.entries_file, line_count, len(row))
                    )
                strain_name = row[0]
                accession = row[1]
                logger.debug("Loading entry {}...".format(accession))
                fasta_filename = fetch_fasta(
                    accession,
                    base_dir=cfg.database_cfg.data_dir,
                    force_download=force_refresh
                )
                yield strain_name, accession, fasta_filename
        logger.info("Found {} records.".format(line_count - 1))
