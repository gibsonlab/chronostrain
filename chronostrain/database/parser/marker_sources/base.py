from abc import ABC, abstractmethod

from chronostrain.model import Marker


class AbstractMarkerSource(ABC):
    @abstractmethod
    def extract_from_primer(
            self,
            marker_id: str, marker_name: str,
            forward: str, reverse: str
    ) -> Marker:
        raise NotImplementedError()

    @abstractmethod
    def extract_subseq(
            self,
            marker_id: str, marker_name: str,
            start_pos: int, end_pos: int, from_negative_strand: bool
    ) -> Marker:
        raise NotImplementedError()

    @abstractmethod
    def extract_from_locus_tag(
            self,
            marker_id: str, marker_name: str,
            locus_tag: str
    ) -> Marker:
        raise NotImplementedError()

    @abstractmethod
    def extract_fasta_record(
            self,
            marker_id: str, marker_name: str, record_id: str, allocate: bool
    ) -> Marker:
        raise NotImplementedError()
