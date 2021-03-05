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

    def compute_log_likelihood(self, fragment: Fragment, read: SequenceRead) -> float:
        return self.match_log_likelihood if fragment.seq == read.seq else self.mismatch_log_likelihood

    def sample_noisy_read(self, fragment: str, metadata: str = "") -> SequenceRead:
        return SequenceRead(fragment, quality=torch.ones(len(fragment))*1000, metadata=metadata)
