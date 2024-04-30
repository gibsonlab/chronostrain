from typing import Union

import numpy as np
from chronostrain.util.sequences import Sequence
from .phred import PhredErrorModel
from .base import SequenceRead


class PairedEndRead(SequenceRead):
    def __init__(self,
                 read_id: str, seq: Sequence, quality: np.ndarray, metadata: str,
                 is_forward: bool):
        super().__init__(read_id, seq, quality, metadata)
        self.is_forward = is_forward
        self.mate_pair: Union[PairedEndRead, None] = None

    @property
    def is_reverse(self) -> bool:
        return not self.is_forward

    def set_mate_pair(self, read: 'PairedEndRead'):
        self.mate_pair = read

    @property
    def has_mate_pair(self) -> bool:
        return self.mate_pair is not None


class PEPhredErrorModel(PhredErrorModel):
    def __init__(self,
                 insertion_error_ll_1: float,
                 deletion_error_ll_1: float,
                 insertion_error_ll_2: float,
                 deletion_error_ll_2: float):
        super().__init__(insertion_error_ll_1, deletion_error_ll_1)
        self.insertion_error_ll_2 = insertion_error_ll_2
        self.deletion_error_ll_2 = deletion_error_ll_2

    def indel_ll(self, read: SequenceRead, insertions: np.ndarray, deletions: np.ndarray):
        if not isinstance(read, PairedEndRead):
            raise Exception("PEErrorModel must use paired-end reads.")

        n_insertions = np.sum(insertions)
        n_deletions = np.sum(deletions)
        if read.is_forward:
            return (n_insertions * self.insertion_error_ll) + (n_deletions * self.deletion_error_ll)
        else:
            return (n_insertions * self.insertion_error_ll_2) + (n_deletions * self.deletion_error_ll_2)
