from typing import List

from chronostrain.model import Marker, Strain
from .base import AbstractStrainDatabaseBackend
from ..error import QueryNotFoundError


class DictionaryBackend(AbstractStrainDatabaseBackend):

    def __init__(self):
        self.strains = {}
        self.markers = {}
        self.markers_to_strains = {}

    def add_strain(self, strain: Strain):
        self.strains[strain.id] = strain
        for marker in strain.markers:
            self.markers[marker.id] = marker
            if not (marker.id in self.markers_to_strains):
                self.markers_to_strains[marker.id] = []
            self.markers_to_strains[marker.id].append(strain)

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
        except KeyError:
            raise QueryNotFoundError("Unable to find marker with id `{}`.".format(marker_id))

    def num_strains(self) -> int:
        return len(self.strains)

    def num_markers(self) -> int:
        return len(self.markers)

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def all_markers(self) -> List[Marker]:
        return list(self.markers.values())

    def get_strains_with_marker(self, marker: Marker) -> List[Strain]:
        try:
            return self.markers_to_strains[marker.id]
        except KeyError:
            return []