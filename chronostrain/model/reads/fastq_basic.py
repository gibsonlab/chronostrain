import numpy as np
from chronostrain.model import Fragment
from chronostrain.model.reads.base import AbstractErrorModel, SequenceRead
from chronostrain.model.reads.basic import RampUpRampDownDistribution
import chronostrain.util.sequences as cseq


class BasicPhredScoreDistribution(RampUpRampDownDistribution):
    """
        An implementation of the quality score ramp-up ramp-down model, where the quality values are PHRED scores.
        ref: https://en.wikipedia.org/wiki/Phred_quality_score
    """

    def __init__(self, length):
        super().__init__(
            length,
            quality_score_values=np.array([10, 20, 30, 40, 50]),
            distribution=np.array([0.05, 0.15, 0.30, 0.25, 0.25])
        )


class BasicFastQErrorModel(AbstractErrorModel):
    """
    A simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicPhredScoreDistribution.
    """
    def __init__(self, read_len=150):
        self.read_len = read_len
        self.q_dist = BasicPhredScoreDistribution(read_len)

    @staticmethod
    def phred_error_prob(q: np.ndarray) -> np.ndarray:
        return np.power(10, -0.1 * q)

    def compute_log_likelihood(self, fragment: Fragment, read: SequenceRead) -> float:
        # NOTE: Ignore quality score distributions (assume negligible/constant likelihood for all q-score vectors.)
        # This only uses phred scores to compute Pr(Read | Fragment, Quality).
        error_prob = self.phred_error_prob(read.quality)
        matches: np.ndarray = (fragment.seq == read.seq)

        p_matches = 1 - error_prob[np.where(matches)]
        p_errors = (1/3) * error_prob[np.where(~matches)]
        return np.log(p_matches).sum() + np.log(p_errors).sum()

    def sample_noisy_read(self, fragment: Fragment, metadata="") -> SequenceRead:
        qvec = self.q_dist.sample_qvec()
        read = SequenceRead(fragment.seq, qvec, metadata=metadata)

        # Random shift by an integer mod 4.
        error_probs = self.phred_error_prob(qvec)
        error_locations: np.ndarray = (np.random.rand(error_probs.shape[0]) < error_probs)  # dtype `bool`

        rand_shift = np.random.randint(low=0, high=4, size=np.sum(error_locations), dtype=cseq.SEQ_DTYPE)
        read.seq[np.where(error_locations)] = np.mod(
            read.seq[np.where(error_locations)] + rand_shift,
            4
        )
        return read
