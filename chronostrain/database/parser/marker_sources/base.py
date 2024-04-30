from abc import ABC, abstractmethod

from chronostrain.model import Marker


class AbstractMarkerSource(ABC):

    @abstractmethod
    def extract_subseq(
            self,
            record_idx: int,
            marker_id: str, marker_name: str,
            start_pos: int, end_pos: int, from_negative_strand: bool
    ) -> Marker:
        raise NotImplementedError()

    @abstractmethod
    def extract_fasta_record(
            self,
            marker_id: str, marker_name: str, record_id: int, allocate: bool
    ) -> Marker:
        raise NotImplementedError()
