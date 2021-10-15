from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import numpy as np

from chronostrain.model import Marker
from chronostrain.util.sequences import SeqType, NucleotideDtype, z4_to_nucleotides

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


@dataclass
class NucleotideInsertion(object):
    insert_pos: int
    insert_len: int
    relative_pos: int
    nucleotide: NucleotideDtype


@dataclass
class NucleotideDeletion(object):
    delete_pos: int


class SequenceReadAlignment(object):
    def __init__(self,
                 read_id: str,
                 sam_path: Path,
                 sam_line_no: int,
                 read_seq: np.ndarray,
                 read_qual: np.ndarray,
                 marker: Marker,
                 read_start: int,
                 read_end: int,
                 marker_start: int,
                 marker_end: int,
                 reverse_complemented: bool,
                 insertions_into_marker: List[NucleotideInsertion],
                 deletions_from_marker: List[NucleotideDeletion]):
        """
        :param read_id: The query read ID.
        :param sam_path: The SAM file that this alignment was parsed from.
        :param sam_line_no: The line number of the samhandler.
        :param read_seq: The query read sequence, minus any hard-clipped regions.
        :param read_qual: The query read quality, minus any hard-clipped regions.
        :param marker: The reference marker.
        :param read_start: The left endpoint of the read at which alignment starts; inclusive.
        :param read_end: The right endpoint of the read at which alignment starts; inclusive.
        :param marker_start: The left endpoint of the marker at which alignment starts; inclusive.
        :param marker_end: The left endpoint of the marker at which alignment starts; inclusive.
        :param reverse_complemented: Indicates whether the read sequence has been reverse-complemented from the original
            query. If so, then the quality is assumed to be reversed from the original query.
        """
        self.read_id: str = read_id
        self.sam_path: Path = sam_path
        self.sam_line_no: int = sam_line_no
        self.id: str = "{}[{},{}]".format(read_id, marker_start, marker_end)
        self.read_seq: np.ndarray = read_seq
        self.read_qual: np.ndarray = read_qual
        self.marker: Marker = marker

        assert (marker_end - marker_start) == (read_end - read_start)
        self.read_start: int = read_start
        self.read_end: int = read_end
        self.marker_start: int = marker_start
        self.marker_end: int = marker_end

        self.marker_frag: SeqType = marker.seq[marker_start:marker_end + 1]

        # Indicates whether the read has been reverse complemented.
        self.reverse_complemented: bool = reverse_complemented

        self.insertions_into_marker: List[NucleotideInsertion] = insertions_into_marker
        self.deletions_from_marker: List[NucleotideDeletion] = deletions_from_marker

    @property
    def read_aligned_section(self) -> Tuple[np.ndarray, np.ndarray]:
        section = slice(self.read_start, self.read_end + 1)
        return self.read_seq[section], self.read_qual[section]

    @property
    def read_seq_nucleotide(self) -> str:
        return z4_to_nucleotides(self.read_seq)

    def __eq__(self, other: 'SequenceReadAlignment') -> bool:
        return self.id == other.id
