from typing import List

from chronostrain.model import Marker, Strain
from .base import AbstractStrainDatabaseBackend, QueryNotFoundError


class DictionaryBackend(AbstractStrainDatabaseBackend):
    def __init__(self):
        self.strains = {}
        self.markers = {}
        self.markers_to_strains = {}

    def add_strain(self, strain: Strain):
        self.strains[strain.id] = strain
        for marker in strain.markers:
            self.markers[marker.name] = marker
            if not (marker.name in self.markers_to_strains):
                self.markers_to_strains[marker.name] = []
            self.markers_to_strains[marker.name].append(strain)

    def get_strain(self, strain_id: str) -> Strain:
        try:
            return self.strains[strain_id]
        except KeyError:
            raise QueryNotFoundError(strain_id)

    def get_marker(self, marker_name: str) -> Marker:
        try:
            return self.markers[marker_name]
        except KeyError:
            raise QueryNotFoundError(marker_name)

    def num_strains(self) -> int:
        return len(self.strains)

    def num_markers(self) -> int:
        return len(self.markers)

    def all_strains(self) -> List[Strain]:
        return list(self.strains.values())

    def all_markers(self) -> List[Marker]:
        return list(self.markers.values())

    def get_strains_with_marker(self, marker_name: str) -> List[Strain]:
        try:
            return self.markers_to_strains[marker_name]
        except KeyError:
            raise QueryNotFoundError(marker_name)
