import math

from chronostrain.model.reads.base import *


class NoiselessErrorModel(AbstractErrorModel):
    """
    A noiseless fragment-to-read error model. By definition, this implemenation ignores quality scores.
    """
    def __init__(self, mismatch_likelihood: float = 1e-20):
        super().__init__()
        self.mismatch_log_likelihood = math.log(mismatch_likelihood) if mismatch_likelihood != 0 else -float("inf")
        self.match_log_likelihood = math.log(1 - mismatch_likelihood)

    def compute_log_likelihood(self,
                               fragment: Fragment,
                               read: SequenceRead,
                               read_reverse_complemented: bool,
                               insertions: np.ndarray,
                               deletions: np.ndarray,
                               read_start_clip: int = 0,
                               read_end_clip: int = 0) -> float:
        if np.sum(insertions) > 0 or np.sum(deletions) > 0:
            return self.mismatch_log_likelihood

        if read_reverse_complemented:
            read_seq = reverse_complement_seq(read.seq)
        else:
            read_seq = read.seq

        _slice = slice(read_start_clip, len(read_seq) - read_end_clip)
        read_seq = read_seq[_slice]

        if np.sum(fragment.seq != read_seq) == 0:
            return self.match_log_likelihood
        else:
            return self.mismatch_log_likelihood

    def sample_noisy_read(self, read_id: str, fragment: Fragment, metadata: str = "") -> SequenceRead:
        return SequenceRead(read_id=read_id, seq=fragment.seq, quality=np.ones(len(fragment))*1000, metadata=metadata)
