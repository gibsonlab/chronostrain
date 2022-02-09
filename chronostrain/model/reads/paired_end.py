from typing import Union

import numpy as np
from .phred import PhredErrorModel
from .base import SequenceRead
from ...util.sequences import SeqType


class PairedEndRead(SequenceRead):
    def __init__(self,
                 read_id: str, seq: Union[str, SeqType], quality: np.ndarray, metadata: str,
                 forward: bool):
        super().__init__(read_id, seq, quality, metadata)
        self.forward = forward

    @property
    def is_forward(self) -> bool:
        return self.forward

    @property
    def is_reverse(self) -> bool:
        return not self.forward


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
            return super().indel_ll(read, insertions, deletions)

        n_insertions = np.sum(insertions)
        n_deletions = np.sum(deletions)
        if read.forward:
            return (n_insertions * self.insertion_error_ll) + (n_deletions * self.deletion_error_ll)
        else:
            return (n_insertions * self.insertion_error_ll_2) + (n_deletions * self.deletion_error_ll_2)
