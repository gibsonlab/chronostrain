from dataclasses import dataclass
from typing import Union, Tuple

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from chronostrain.util.sequences import Sequence


@dataclass
class MarkerMetadata:
    parent_strain: str
    parent_seq: str

    def __repr__(self):
        return "MarkerMetadata[{}:{}]".format(self.parent_strain, self.parent_seq)

    def __str__(self):
        return self.__repr__()


@dataclass
class Marker:
    id: str  # A unique identifier.
    name: str  # A human-readable name.
    seq: Sequence
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

    def to_seqrecord(self, description: str = "") -> SeqRecord:
        return SeqRecord(
            Seq(self.seq.nucleotides()),
            id=f"{self.name}|{self.id}",
            description=description
        )

    @staticmethod
    def parse_seqrecord_id(record_id: str) -> Tuple[str, str]:
        name, marker_id = record_id.split("|")
        return name, marker_id
