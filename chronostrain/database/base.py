from pathlib import Path
from abc import abstractmethod, ABCMeta
from typing import List

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.model.bacteria import Strain
from chronostrain.config import cfg
from . import logger


class AbstractStrainDatabase(metaclass=ABCMeta):
    def __init__(self, force_refresh: bool = False):
        self.__load__(force_refresh=force_refresh)
        self.multifasta_file: Path = cfg.database_cfg.data_dir / 'all_markers.fasta'
        self._save_markers_to_multifasta(
            force_refresh=force_refresh
        )

    @abstractmethod
    def __load__(self, force_refresh: bool = False):
        """
        Loads the database. Automatically called by the constructor __init__.
        :param force_refresh: If true, database should refresh entire index; if necessary, should
        re-download relevant files.
        :return:
        """
        pass

    @abstractmethod
    def get_strain(self, strain_id: str) -> Strain:
        pass

    @abstractmethod
    def all_strains(self) -> List[Strain]:
        pass

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return [self.get_strain(s_id) for s_id in strain_ids]

    @abstractmethod
    def num_strains(self) -> int:
        pass

    def strain_markers_to_fasta(self, strain_id: str, out_path: Path, file_mode: str = "w"):
        strain = self.get_strain(strain_id)
        records = []
        for marker in strain.markers:
            records.append(marker.to_seqrecord(description="Strain:{}".format(strain.metadata.ncbi_accession)))
        with open(out_path, file_mode) as out_file:
            SeqIO.write(records, out_file, "fasta")

    def _save_markers_to_multifasta(self, force_refresh: bool = True):
        """
        Save all markers to a single, concatenated multi-fasta file.
        The file will be automatically re-populated if force_refresh is True, or if the existing file is stale (e.g.
        there exists a marker whose last-modified timestamp is later than the existing file's.)
        """
        self.multifasta_file.resolve().parent.mkdir(exist_ok=True, parents=True)

        def _generate():
            self.multifasta_file.unlink(missing_ok=True)
            for strain in self.all_strains():
                self.strain_markers_to_fasta(strain.id, self.multifasta_file, "a+")

        if force_refresh:
            logger.debug("Forcing re-creation of multi-fasta file.")
            _generate()
        elif self.multifasta_file.exists():
            if self._multifasta_is_stale():
                logger.debug("Multi-fasta file exists, but is stale. Re-creating.")
                _generate()
            else:
                logger.debug("Multi-fasta file already exists. Skipping creation.")
        else:
            _generate()

        logger.debug("Multi-fasta file: {}".format(self.multifasta_file))

    def _multifasta_is_stale(self):
        for strain in self.all_strains():
            for marker in strain.markers:
                if marker.metadata.file_path.stat().st_mtime > self.multifasta_file.stat().st_mtime:
                    return True
        return False


class StrainEntryError(BaseException):
    pass


class StrainNotFoundError(BaseException):
    def __init__(self, strain_id):
        self.strain_id = strain_id
        super().__init__("Strain id `{}` not found in database.".format(strain_id))
