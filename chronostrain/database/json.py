import json
from typing import List

from chronostrain.database.base import AbstractStrainDatabase, SubsequenceLoader, StrainEntryError
from chronostrain.model.bacteria import Marker, MarkerMetadata, Strain
from chronostrain.util.io.fetch_genomes import fetch_filenames
from chronostrain.util.io.logger import logger


def parse_strain_info(json_dict):
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


class JSONStrainDatabase(AbstractStrainDatabase):
    """
    A Simple implementation that treats each complete strain genome and optional specified subsequences as markers.
    """

    def __init__(self, json_refs):
        """
        :param json_refs: JSON file specifying accession numbers and marker locus tags.
        """
        self.strains = dict()  # accession -> Strain
        self.json_refs = json_refs
        super().__init__()

    def __load__(self):
        for strain_name, strain_accession, strain_markers in self.strain_entries():
            fasta_filename, genbank_filename = fetch_filenames(strain_accession)
            sequence_loader = SubsequenceLoader(fasta_filename, genbank_filename, strain_markers)

            genome = sequence_loader.get_full_genome()
            markers = []
            for subsequence_data in sequence_loader.get_marker_subsequences():
                markers.append(Marker(
                    name=subsequence_data.ID,
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
        return self.strains[strain_id]

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def strain_entries(self):
        """
        Read JSON file, and download FASTA from accessions if doesn't exist.
        :return: a dictionary mapping accessions to strain-accession-filename-subsequences
                 wrappers.
        """
        with open(self.json_refs, "rb") as f:
            for strain_dict in json.load(f):
                yield parse_strain_info(strain_dict)
