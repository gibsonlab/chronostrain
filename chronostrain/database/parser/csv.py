import csv
from pathlib import Path
from typing import Iterator, Tuple, Optional

from Bio import SeqIO

from chronostrain.config import cfg
from chronostrain.config.logging import create_logger
from chronostrain.model import Strain, Marker, MarkerMetadata
from chronostrain.model.bacteria import StrainMetadata
from chronostrain.util.sequences import nucleotides_to_z4
from chronostrain.util.entrez import fetch_fasta
from .base import AbstractDatabaseParser, StrainDatabaseParseError

logger = create_logger(__name__)


class CSVParser(AbstractDatabaseParser):
    def __init__(self,
                 entries_file: Path,
                 force_refresh: bool = False,
                 trim_debug: Optional[int] = None,
                 load_full_genomes: bool = False
                 ):
        self.entries_file = entries_file
        self.force_refresh = force_refresh
        self.trim_debug = trim_debug
        self.load_full_genomes = load_full_genomes

    def strains(self) -> Iterator[Strain]:
        logger.info("Loading from CSV marker database file {}.".format(self.entries_file))
        for strain_name, accession, strain_fasta_path in self._strain_entries(self.force_refresh):
            record = next(SeqIO.parse(strain_fasta_path, "fasta"))
            genome = str(record.seq)
            if self.trim_debug is not None:
                markers = [
                    Marker(
                        name="Genome_{}[{}:{}]".format(accession, 0, self.trim_debug),
                        id="GENOME[{}]".format(accession),
                        seq=nucleotides_to_z4(genome[:self.trim_debug]),
                        metadata=MarkerMetadata(parent_accession=accession,
                                                file_path=strain_fasta_path),
                        canonical=True
                    )
                ]
            else:
                markers = [
                    Marker(
                        name="Genome_{}".format(accession),
                        id="GENOME[{}]".format(accession),
                        seq=nucleotides_to_z4(genome),
                        metadata=MarkerMetadata(parent_accession=accession,
                                                file_path=strain_fasta_path),
                        canonical=True
                    )
                ]  # Each genome's marker is its own genome.
            yield Strain(
                id=accession,
                markers=markers,
                metadata=StrainMetadata(
                    ncbi_accession=accession,
                    genus="",
                    species=strain_name,
                    source_path=strain_fasta_path
                )
            )

    def _strain_entries(self, force_refresh: bool) -> Iterator[Tuple[str, str, Path]]:
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
                    raise StrainDatabaseParseError("{} -- Expected 2 tokens on line {}, got {}.".format(
                        self.entries_file, line_count, len(row))
                    )
                strain_name = row[0]
                accession = row[1]
                logger.debug("Loading entry {}...".format(accession))
                fasta_path = fetch_fasta(
                    accession,
                    base_dir=cfg.database_cfg.data_dir,
                    force_download=force_refresh
                )
                yield strain_name, accession, fasta_path
        logger.info("Found {} records.".format(line_count - 1))
