from abc import abstractmethod
from typing import List

from chronostrain.model import Strain, Marker


class QueryNotFoundError(BaseException):
    def __init__(self, query):
        super().__init__("Query `{}` not found in database.".format(query))


class AbstractStrainDatabaseBackend(object):
    @abstractmethod
    def add_strain(self, strain: Strain):
        pass

    @abstractmethod
    def get_strain(self, strain_id: str) -> Strain:
        pass

    @abstractmethod
    def get_marker(self, marker_name: str) -> Marker:
        pass

    @abstractmethod
    def num_strains(self) -> int:
        pass

    @abstractmethod
    def num_markers(self) -> int:
        pass

    @abstractmethod
    def all_strains(self) -> List[Strain]:
        pass

    @abstractmethod
    def all_markers(self) -> List[Marker]:
        pass

    @abstractmethod
    def get_strains_with_marker(self, marker_name: str) -> List[Strain]:
        pass
