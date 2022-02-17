from pathlib import Path
from dataclasses import dataclass
from typing import List, Union, Iterator, Tuple

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.config.logging import create_logger
from chronostrain.util.sequences import SeqType, z4_to_nucleotides
logger = create_logger(__name__)


@dataclass
class MarkerMetadata:
    parent_strain: str
    file_path: Union[Path, None]
    
    def __repr__(self):
        if self.file_path is not None:
            return "MarkerMetadata[{}:{}]".format(self.parent_strain, self.file_path)
        else:
            return "MarkerMetadata[{}]".format(self.parent_strain)
        
    def __str__(self):
        return self.__repr__()


@dataclass
class StrainMetadata:
    chromosomes: List[str]
    scaffolds: List[str]
    genus: str
    species: str
    total_len: int


@dataclass
class Marker:
    id: str  # A unique identifier.
    name: str  # A human-readable name.
    seq: SeqType
    canonical: bool
    metadata: Union[MarkerMetadata, None] = None

    def __repr__(self):
        return "Marker[{}]".format(self.id)

    def __str__(self):
        return "Marker[{}]".format(self.id)

    def __len__(self):
        return len(self.seq)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    @property
    def nucleotide_seq(self) -> str:
        """
        The ACGT nucleotide sequence of this marker.
        """
        return z4_to_nucleotides(self.seq)

    @property
    def is_canonical(self) -> bool:
        return self.canonical

    def to_seqrecord(self, description: str = "") -> SeqRecord:
        return SeqRecord(
            Seq(self.nucleotide_seq),
            id="{}|{}|{}".format(self.metadata.parent_strain, self.name, self.id),
            description=description
        )

    @staticmethod
    def parse_seqrecord_id(record_id: str) -> Tuple[str, str, str]:
        parent_strain, name, marker_id = record_id.split("|")
        return parent_strain, name, marker_id


@dataclass
class Strain:
    id: str  # Typically, ID is the accession number.
    name: str
    markers: List[Marker]
    metadata: Union[StrainMetadata, None] = None

    def __repr__(self):
        return "{}({}:{})".format(
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


class Population:
    def __init__(self, strains: List[Strain]):
        """
        :param strains: a list of Strain instances.
        """

        if not all([isinstance(s, Strain) for s in strains]):
            raise ValueError("All elements in strains must be Strain instances")

        self.strains = list(strains)  # A list of Strain objects.
        self.all_markers = {
            marker
            for strain in strains
            for marker in strain.markers
        }

    def __hash__(self):
        """
        Returns a hashed representation of the strain collection.
        :return:
        """
        return "[{}]".format(
            ",".join(strain.__repr__() for strain in self.strains)
        ).__hash__()

    def __repr__(self):
        return self.strains.__repr__()

    def __str__(self):
        return "[{}]".format(
            ",".join(strain.__str__() for strain in self.strains)
        )

    def num_strains(self) -> int:
        return len(self.strains)

    def markers_iterator(self) -> Iterator[Marker]:
        for strain in self.strains:
            for marker in strain.markers:
                yield marker

    def contains_marker(self, marker: Marker) -> bool:
        return marker in self.all_markers
