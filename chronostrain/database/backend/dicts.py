from collections import defaultdict
from typing import List, Iterator, Dict

from chronostrain.model import Marker, Strain
from .base import AbstractStrainDatabaseBackend
from ..error import QueryNotFoundError


class DictionaryBackend(AbstractStrainDatabaseBackend):

    def __init__(self):
        self.strain_order: List[Strain] = []
        self.strains: Dict[str, Strain] = {}
        self.markers: Dict[str, Marker] = {}
        self.markers_to_strains: Dict[str, List[Strain]] = {}
        self.markers_by_name: Dict[str, List[Marker]] = defaultdict(list)

    def add_strains(self, strains: Iterator[Strain]):
        for strain in strains:
            self.strains[strain.id] = strain
            for marker in strain.markers:
                self.markers[marker.id] = marker
                if not (marker.id in self.markers_to_strains):
                    self.markers_to_strains[marker.id] = []
                self.markers_to_strains[marker.id].append(strain)
                self.markers_by_name[marker.name].append(marker)
            self.strain_order.append(strain)

    def get_strain(self, strain_id: str) -> Strain:
        try:
            return self.strains[strain_id]
        except KeyError:
            raise QueryNotFoundError("Unable to find strain with id `{}`.".format(strain_id))

    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        return [self.get_strain(strain_id) for strain_id in strain_ids]

    def get_marker(self, marker_id: str) -> Marker:
        try:
            return self.markers[marker_id]
        except KeyError as e:
            raise QueryNotFoundError("Unable to find marker with id `{}`.".format(marker_id)) from e

    def num_strains(self) -> int:
        return len(self.strains)

    def num_markers(self) -> int:
        return len(self.markers)

    def all_strains(self) -> List[Strain]:
        return self.strain_order

    def all_markers(self) -> List[Marker]:
        return list(self.markers.values())

    def get_strain_with_marker(self, marker: Marker) -> Strain:
        return self.markers_to_strains[marker.id][0]

    def get_markers_by_name(self, marker_name: str) -> List[Marker]:
        try:
            return self.markers_by_name[marker_name]
        except KeyError:
            return []

    def signature(self) -> str:
        import hashlib
        return "DICT={}".format(
            hashlib.sha256("|".join(
                "{}:{}".format(
                    strain.id,
                    "+".join(
                        marker.id
                        for marker in strain.markers
                    )
                )
                for strain in self.strain_order
            ).encode()).hexdigest()
        )
