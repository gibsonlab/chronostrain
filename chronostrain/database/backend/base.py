from abc import abstractmethod
from typing import List, Iterator

from chronostrain.model import Strain, Marker


class AbstractStrainDatabaseBackend(object):
    @abstractmethod
    def add_strains(self, strains: Iterator[Strain]):
        """
        Registers a strain into this database.
        """
        pass

    @abstractmethod
    def get_strain(self, strain_id: str) -> Strain:
        """
        Look up and retrieve a Strain instance associated with the id.
        """
        pass

    @abstractmethod
    def get_strains(self, strain_ids: List[str]) -> List[Strain]:
        """
        For each strain id, look up and retrieve a Strain instance.
        """
        pass

    @abstractmethod
    def get_marker(self, marker_id: str) -> Marker:
        """
        Look up and retrieve a Marker instance associated with the name.
        """
        pass

    @abstractmethod
    def num_strains(self) -> int:
        """
        Count the total number of registered strains.
        """
        pass

    @abstractmethod
    def num_markers(self) -> int:
        """
        Count the total number of registered markers.
        """
        pass

    @abstractmethod
    def all_strains(self) -> List[Strain]:
        """
        Retrieve all strains registered into the database.
        """
        pass

    @abstractmethod
    def all_markers(self) -> List[Marker]:
        """
        Retrieve all markers registered into the database.
        """
        pass

    @abstractmethod
    def get_strains_with_marker(self, marker: Marker) -> List[Strain]:
        """
        Retrieve all strains that contains this marker.
        """
        pass

    def get_markers_by_name(self, marker_name: str) -> List[Marker]:
        """
        Retrieve all strains with name matching the query.
        """
        pass

    def get_canonical_marker(self, marker_name: str) -> Marker:
        """
        Retrieve the canonical version of the specified marker.
        """
        pass

    def signature(self) -> str:
        pass

    def all_canonical_markers(self) -> List[Marker]:
        pass

    def num_canonical_markers(self) -> int:
        pass
