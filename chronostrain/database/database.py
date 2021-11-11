from collections import defaultdict
from pathlib import Path
from typing import List, Union, Optional

from Bio import SeqIO

from chronostrain.model import Strain, Marker
from .parser import AbstractDatabaseParser, JSONParser, CSVParser
from .backend import AbstractStrainDatabaseBackend, DictionaryBackend
from .error import QueryNotFoundError
from .. import cfg, create_logger

logger = create_logger(__name__)


class StrainDatabase(object):
    def __init__(self,
                 parser: AbstractDatabaseParser,
                 backend: AbstractStrainDatabaseBackend,
                 force_refresh: bool = False):
        self.backend = backend
        self.marker_multifasta_file = cfg.database_cfg.data_dir / 'all_markers.fasta'

        for strain in parser.strains():
            backend.add_strain(strain)
            logger.debug(f"Added strain entry {strain.id} to database.")

        self._save_markers_to_multifasta(force_refresh=force_refresh)

    def get_strain(self, strain_id: str) -> Strain:
        return self.backend.get_strain(strain_id)

    def all_strains(self) -> List[Strain]:
        return self.backend.all_strains()

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return self.backend.get_strains(strain_ids)

    def all_markers(self) -> List[Marker]:
        return self.backend.all_markers()

    def get_marker(self, marker_id: str) -> Marker:
        return self.backend.get_marker(marker_id)

    def num_strains(self) -> int:
        return self.backend.num_strains()

    def num_markers(self) -> int:
        return self.backend.num_markers()

    def get_strains_with_marker(self, marker: Marker) -> List[Strain]:
        return self.backend.get_strains_with_marker(marker)

    def best_matching_strain(self, query_markers: List[Marker]) -> Strain:
        strain_num_hits = defaultdict(int)
        for marker in query_markers:
            for strain in self.get_strains_with_marker(marker):
                strain_num_hits[strain.id] += 1

        if len(strain_num_hits) == 0:
            raise QueryNotFoundError("No available strains with any of query markers.")

        highest_n_hits = max(strain_num_hits.values())
        best_hits = []
        for strain_id, n_hits in strain_num_hits.items():
            if n_hits == highest_n_hits:
                best_hits.append(strain_id)

        if len(best_hits) > 1:
            logger.warning("Found multiple hits ({}) for query marker set. Returning the first hit only.".format(
                best_hits
            ))

        return self.get_strain(best_hits[0])

    @property
    def multifasta_file(self) -> Path:
        return self.marker_multifasta_file

    def _save_markers_to_multifasta(self, force_refresh: bool = True):
        """
        Save all markers to a single, concatenated multi-fasta file.
        The file will be automatically re-populated if force_refresh is True, or if the existing file is stale (e.g.
        there exists a marker whose last-modified timestamp is later than the existing file's.)
        """
        self.multifasta_file.resolve().parent.mkdir(exist_ok=True, parents=True)

        def _generate():
            self.multifasta_file.unlink(missing_ok=True)
            with open(self.multifasta_file, "w"):
                pass
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

    def strain_markers_to_fasta(self, strain_id: str, out_path: Path, file_mode: str = "w"):
        strain = self.get_strain(strain_id)
        records = []
        for marker in strain.markers:
            records.append(marker.to_seqrecord(description=""))
        with open(out_path, file_mode) as out_file:
            SeqIO.write(records, out_file, "fasta")


class JSONStrainDatabase(StrainDatabase):
    def __init__(self,
                 entries_file: Union[str, Path],
                 marker_max_len: int,
                 force_refresh: bool = False,
                 load_full_genomes: bool = False):
        if isinstance(entries_file, str):
            entries_file = Path(entries_file)
        parser = JSONParser(entries_file,
                            marker_max_len,
                            force_refresh,
                            load_full_genomes=load_full_genomes)
        backend = DictionaryBackend()
        super().__init__(parser, backend)


class SimpleCSVStrainDatabase(StrainDatabase):
    def __init__(self,
                 entries_file: Union[str, Path],
                 trim_debug: Optional[int] = None,
                 force_refresh: bool = False,
                 load_full_genomes: bool = False):
        if isinstance(entries_file, str):
            entries_file = Path(entries_file)
        parser = CSVParser(entries_file,
                           force_refresh,
                           trim_debug,
                           load_full_genomes=load_full_genomes)
        backend = DictionaryBackend()
        super().__init__(parser, backend)
