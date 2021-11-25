from abc import abstractmethod
from typing import Tuple, List, Iterator
import numpy as np

from chronostrain.model import Marker, Strain, SequenceRead, MarkerMetadata
from chronostrain.util.sequences import *


class AbstractMarkerVariant(Marker):
    def __init__(self,
                 id: str,
                 name: str,
                 seq: SeqType,
                 base_marker: Marker,
                 metadata: MarkerMetadata):
        super().__init__(id=id, name=name, seq=seq, metadata=metadata, canonical=False)
        self.base_marker = base_marker

    @abstractmethod
    def subseq_from_read(
            self,
            read: SequenceRead
    ) -> Iterator[Tuple[SeqType, bool, np.ndarray, np.ndarray, int, int]]:
        """
        :param read:
        :return: A tuple of numpy arrays specifying:
            1) The fragment subsequence corresponding to this read which comes from the marker variant.
            2) A boolean indicating whether or not to reverse complement.
            3) An "insertions" array of boolean values, specifying insertions INTO the marker.
            4) A "deletions" array of boolean values, specifying deletions FROM the marker.
            5) The number of bases to clip from the start.
            6) The number of bases to clip from the end.
        """
        pass

    @abstractmethod
    def subseq_from_pairwise_aln(self, aln):
        pass


class StrainVariant(Strain):
    def __init__(self,
                 base_strain: Strain,
                 id: str,
                 variant_markers: List[AbstractMarkerVariant]
                 ):
        super().__init__(
            id=id,
            markers=variant_markers,
            metadata=base_strain.metadata
        )
        self.variant_markers = variant_markers
        self.base_strain = base_strain
