import bz2
import os
import pickle
from typing import Tuple, List

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.config import logger, cfg
from chronostrain.database.base import AbstractStrainDatabase, StrainNotFoundError
from chronostrain.model.bacteria import Marker, MarkerMetadata, Strain, StrainMetadata


class MetaphlanDatabase(AbstractStrainDatabase):
    """
    An implementation which parses a metaphlan (v3.0) database into strain/marker entries.
    """
    def __init__(self,
                 basepath: str,
                 strain_universe: str = "",
                 prune_empty: bool = True):
        """
        :param basepath: The basepath of the metaphlan database. Example: "metaphlan_db/mpa_v30_CHOCOPhlAn_201901".
        This implementation expects a .pkl and a .fna.bz2 file of the specified basename.
        """
        if len(strain_universe) > 0:
            self.strain_universe = {
                token.strip()
                for token in strain_universe.split(",")
            }

            print("STRAIN UNIVERSE: {}".format(self.strain_universe))
        else:
            self.strain_universe = None
        self.pickle_path = "{}.pkl".format(basepath)
        self.marker_seq_path = "{}.fna.bz2".format(basepath)
        self.id_to_strains = dict()  # Clade name -> Strain
        self.prune_empty = prune_empty
        super().__init__()

        self.marker_multifasta_path = os.path.join(cfg.database_cfg.data_dir, "usable_markers.fna")
        self.save_markers_to_multifasta(filepath=self.marker_multifasta_path)
        logger.debug("Multi-fasta file: {}".format(self.marker_multifasta_path))

    def __load__(self):
        logger.info("Loading from Metaphlan database pickle {}.".format(self.pickle_path))
        metaphlan_db: dict = pickle.load(bz2.open(self.pickle_path, 'r'))
        self._load_strains(metaphlan_db)
        self._load_markers(metaphlan_db)

        if self.strain_universe is not None:
            self.id_to_strains = {
                strain_id: self.get_strain(strain_id)
                for strain_id in self.strain_universe
            }
            logger.debug("Strain universe restricted to {}.".format(self.strain_universe))
        elif self.prune_empty:
            n_empty_strains = 0
            for strain_id, strain in list(self.id_to_strains.items()):
                if len(strain.markers) == 0:
                    del self.id_to_strains[strain_id]
                    n_empty_strains += 1
            if n_empty_strains > 0:
                logger.debug("Pruned {} strains with zero specified markers.".format(n_empty_strains))

    def _load_strains(self, metaphlan_db: dict):
        for ncbi_assembly_identifier, species, genus, genome_length in self.taxonomy_entries(metaphlan_db):
            self.id_to_strains[ncbi_assembly_identifier] = Strain(
                id=ncbi_assembly_identifier,
                markers=[],
                genome_length=genome_length,
                metadata=StrainMetadata(
                    ncbi_accession=ncbi_assembly_identifier,
                    file_path=os.path.join(cfg.database_cfg.data_dir, "{}.fna".format(ncbi_assembly_identifier)),
                    genus=genus,
                    species=species
                )
            )

    def _load_markers(self, metaphlan_db: dict):
        """
        Loads the Markers using the 'markers' section of the database.
        Assumes that load_strains() has already been invoked.
        :param metaphlan_db:
        :return:
        """
        metaphlan_marker_dict = metaphlan_db['markers']
        with bz2.open(self.marker_seq_path, "rt") as marker_seq_file:
            n_markers_skipped = 0
            for record in SeqIO.parse(marker_seq_file, "fasta"):
                marker_id = record.id
                seq = str(record.seq)
                gene_id = marker_id.split(":")[-1]

                if marker_id not in metaphlan_marker_dict:
                    # logger.debug("Skipping marker id {}".format(marker_id))
                    n_markers_skipped += 1
                    continue
                metaphlan_marker_entry = metaphlan_marker_dict[marker_id]
                strain_ext_seqids = metaphlan_marker_entry['ext']

                # No need to create this marker instance if no relevant strain contains it.
                if len(strain_ext_seqids) == 0:
                    continue

                marker_instance = Marker(
                    name=marker_id,
                    seq=seq,
                    metadata=MarkerMetadata(
                        gene_id=gene_id,
                        file_path=self.marker_seq_path
                    )
                )

                for seq_id in strain_ext_seqids:
                    if seq_id not in self.id_to_strains:
                        # logger.debug("Skipping EXT seqid {} (marker_id={})".format(seq_id, marker_id))
                        continue
                    strain = self.id_to_strains[seq_id]
                    strain.markers.append(marker_instance)
            logger.debug("Skipped {} marker sequence entries.".format(n_markers_skipped))

    @staticmethod
    def taxonomy_entries(metaphlan_db: dict) -> Tuple[str, str, str, int]:
        """
        Helper for _load_strains(). Iterates through the 'taxonomy' dictionary and parses each 7-level taxonomy string.
        :param metaphlan_db:
        :return:
        """
        taxonomies = metaphlan_db['taxonomy']
        n_skipped_entries = 0
        for taxonomy_levels, (ncbi_levels, genome_length) in taxonomies.items():
            taxonomy_tokens = taxonomy_levels.split("|")
            if len(taxonomy_tokens) == 8:
                ncbi_assembly_identifier = remove_prefix(taxonomy_tokens[-1])
                species = remove_prefix(taxonomy_tokens[-2])
                genus = remove_prefix(taxonomy_tokens[-3])
                yield ncbi_assembly_identifier, species, genus, genome_length
            else:
                n_skipped_entries += 1

        logger.debug("Parsed {} strain entries; skipped {}.".format(
            len(taxonomies),
            n_skipped_entries
        ))

    def get_multifasta_file(self) -> str:
        return self.marker_multifasta_path

    def num_strains(self) -> int:
        return len(self.id_to_strains)

    def get_strain(self, strain_id: str) -> Strain:
        if strain_id not in self.id_to_strains:
            raise StrainNotFoundError(strain_id)

        return self.id_to_strains[strain_id]

    def all_strains(self) -> List[Strain]:
        return [strain for _, strain in self.id_to_strains.items()]

    def save_markers_to_multifasta(self,
                                   filepath: str,
                                   check_exists: bool = True):
        if check_exists and os.path.exists(filepath):
            logger.debug("Multi-fasta file already exists; skipping creation.")
        else:
            records = []
            for strain in self.all_strains():
                for marker in strain.markers:
                    records.append(
                        SeqRecord(Seq(marker.seq),
                                  id=marker.name,
                                  description="Strain:{}".format(strain.metadata.ncbi_accession))
                    )
            SeqIO.write(records, filepath, "fasta")


def remove_prefix(tax_token: str) -> str:
    """
    Removes the level identifier of specified token, e.g. "g__Lactobacillus" -> "Lactobacillus".
    :param tax_token:
    :return:
    """
    return tax_token[3:]
