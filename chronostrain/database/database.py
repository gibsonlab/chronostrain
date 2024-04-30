from collections import defaultdict
from pathlib import Path
from typing import List, Set

from chronostrain.model import Strain, Marker
from .backend import AbstractStrainDatabaseBackend
from .error import QueryNotFoundError

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class StrainDatabase(object):
    def __init__(self,
                 backend: AbstractStrainDatabaseBackend,
                 data_dir: Path,
                 name: str):
        self.backend = backend
        self.name = name
        logger.info("Instantiating database `{}`.".format(name))
        self.work_dir = self.database_named_dir(data_dir, name)

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

    def get_strain_with_marker(self, marker: Marker) -> Strain:
        return self.backend.get_strain_with_marker(marker)

    def best_matching_strain(self, query_markers: List[Marker]) -> Strain:
        strain_num_hits = defaultdict(int)
        for marker in query_markers:
            strain = self.get_strain_with_marker(marker)
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
