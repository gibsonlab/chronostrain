from dataclasses import dataclass
from typing import List, Union
from .marker import Marker


@dataclass
class StrainMetadata:
    chromosomes: List[str]
    scaffolds: List[str]
    genus: str
    species: str
    total_len: int
    cluster: List[str]


@dataclass
class Strain:
    id: str  # Typically, ID is the accession number.
    name: str
    markers: List[Marker]
    metadata: Union[StrainMetadata, None] = None

    def __repr__(self):
        return "Strain[{}({}:{})]".format(
            self.__class__.__name__,
            self.id,
            self.markers.__repr__()
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.id
        )

    def __hash__(self):
        return hash(self.id)

    def num_marker_frags(self, frag_len: int) -> int:
        return sum(
            len(marker) - frag_len + 1 for marker in self.markers
        )
