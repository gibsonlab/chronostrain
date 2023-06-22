from collections import defaultdict
from pathlib import Path
from typing import List, Set

from Bio import SeqIO

from chronostrain.model import Strain, Marker
from .backend import AbstractStrainDatabaseBackend
from .error import QueryNotFoundError

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class StrainDatabase(object):
    def __init__(self,
                 backend: AbstractStrainDatabaseBackend,
                 data_dir: Path,
                 name: str,
                 force_refresh: bool = False):
        self.backend = backend
        self.name = name
        logger.info("Instantiating database `{}`.".format(name))

        self.work_dir = self.database_named_dir(data_dir, name)
        self.marker_multifasta_file = self.work_dir / f'all_markers.fasta'
        self.save_markers_to_multifasta(force_refresh=force_refresh)

    @property
    def multifasta_file(self) -> Path:
        return self.marker_multifasta_file

    @property
    def signature(self) -> str:
        return self.backend.signature()

    @staticmethod
    def database_named_dir(data_dir, db_name):
        return data_dir / f'__{db_name}_'

    def get_strain(self, strain_id: str) -> Strain:
        return self.backend.get_strain(strain_id)

    def all_strains(self) -> List[Strain]:
        return self.backend.all_strains()

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return self.backend.get_strains(strain_ids)

    def all_markers(self) -> List[Marker]:
        return self.backend.all_markers()

    def all_marker_names(self) -> Set[str]:
        name_set = set()
        for marker in self.all_markers():
            name_set.add(marker.name)
        return name_set

    def get_marker(self, marker_id: str) -> Marker:
        return self.backend.get_marker(marker_id)

    def get_markers_by_name(self, marker_name: str) -> List[Marker]:
        return self.backend.get_markers_by_name(marker_name)

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
            raise QueryNotFoundError("No available strains with any of query markers: [{}]".format(
                ",".join(m.id for m in query_markers)
            ))

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

    def save_markers_to_multifasta(self, force_refresh: bool = True):
        """
        Save all markers to a single, concatenated multi-fasta file.
        The file will be automatically re-populated if force_refresh is True, or if the existing file is stale (e.g.
        there exists a marker whose last-modified timestamp is later than the existing file's.)
        """
        self.multifasta_file.resolve().parent.mkdir(exist_ok=True, parents=True)

        def _generate():
            if self.multifasta_file.exists():
                self.multifasta_file.unlink()

            with open(self.multifasta_file, "w"):
                pass
            self.markers_to_fasta(self.multifasta_file, "a+")

        if force_refresh:
            logger.debug("Forcing re-creation of multi-fasta file.")
            _generate()
        elif self.multifasta_file.exists():
            # TODO: provide a more efficient implementation of staleness checking.
            # if self._multifasta_is_stale():
            #     logger.debug("Multi-fasta file exists, but is stale. Re-creating.")
            #     _generate()
            # else:
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

    def markers_to_fasta(self, out_path: Path, file_mode: str = "w"):
        records = []
        for marker in self.backend.all_markers():
            records.append(marker.to_seqrecord(description=""))
        with open(out_path, file_mode) as out_file:
            SeqIO.write(records, out_file, "fasta")

    def get_canonical_marker(self, marker_name: str) -> Marker:
        return self.backend.get_canonical_marker(marker_name)

    def all_canonical_markers(self) -> List[Marker]:
        return self.backend.all_canonical_markers()

    def num_canonical_markers(self) -> int:
        return self.backend.num_canonical_markers()
