from abc import abstractmethod
from typing import Tuple, List, Iterator
import numpy as np

from chronostrain.model import Marker, Strain, SequenceRead, MarkerMetadata
from chronostrain.util.sequences import *


class AbstractMarkerVariant(Marker):
    def __init__(self, id: str, name: str, seq: SeqType, base_marker: Marker, metadata: MarkerMetadata):
        super().__init__(id=id, name=name, seq=seq, metadata=metadata)
        self.base_marker = base_marker

    @abstractmethod
    def subseq_from_read(self, read: SequenceRead) -> Iterator[Tuple[SeqType, np.ndarray, np.ndarray]]:
        """
        :param read:
        :return: A triple of numpy arrays specifying:
            1) The fragment subsequence corresponding to this read which comes from the marker variant.
            2) An "insertions" array of boolean values, specifying insertions INTO the marker.
            3) A "deletions" array of boolean values, specifying deletions FROM the marker.
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
