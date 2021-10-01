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


class PhredErrorModel(AbstractErrorModel):
    """
    A simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicPhredScoreDistribution.
    """
    def __init__(self, read_len=150):
        self.read_len = read_len
        self.q_dist = BasicPhredScoreDistribution(read_len)

    def compute_log_likelihood(self, fragment: Fragment, read: SequenceRead) -> float:
        """
        Uses phred scores to compute Pr(Read | Fragment, Quality).
        """
        error_log10_prob = -0.1 * read.quality
        matches: np.ndarray = (fragment.seq == read.seq) & (read.quality > 0)
        # TODO: check whether this filters out `N`s.
        mismatches: np.ndarray = (fragment.seq != read.seq) & (read.quality > 0)

        """
        Phred model: Pr(measured base = 'A', true base = 'G' | q) = ( 1/3 * 10^{-q/10} )
        """
        log_p_errors = -np.log(3) + np.log(10) * error_log10_prob[np.where(mismatches)]
        log_p_matches = np.log(1 - np.power(10, error_log10_prob[np.where(matches)]))
        return log_p_matches.sum() + log_p_errors.sum()

    def sample_noisy_read(self, read_id: str, fragment: Fragment, metadata="") -> SequenceRead:
        qvec = self.q_dist.sample_qvec()
        read = SequenceRead(read_id=read_id, seq=fragment.seq, quality=qvec, metadata=metadata)

        # Random shift by an integer mod 4.
        error_probs = np.power(10, -0.1 * qvec)
        error_locations: np.ndarray = (np.random.rand(error_probs.shape[0]) < error_probs)  # dtype `bool`

        rand_shift = np.random.randint(low=0, high=4, size=np.sum(error_locations), dtype=cseq.SEQ_DTYPE)
        read.seq[np.where(error_locations)] = np.mod(
            read.seq[np.where(error_locations)] + rand_shift,
            4
        )
        return read
