import numpy as np
from chronostrain.model import Fragment
from chronostrain.model.reads.base import AbstractErrorModel, SequenceRead
from chronostrain.model.reads.basic import RampUpRampDownDistribution
from chronostrain.util.sequences import NucleotideDtype, bytes_N, AllocatedSequence


class BasicPhredScoreDistribution(RampUpRampDownDistribution):
    """
        An implementation of the quality score ramp-up ramp-down model, where the quality values are PHRED scores.
        ref: https://en.wikipedia.org/wiki/Phred_quality_score
    """

    def __init__(self, length: int):
        super().__init__(
            length=length,
            quality_score_values=np.array([10, 20, 30, 40, 50]),
            distribution=np.array([0.05, 0.15, 0.30, 0.25, 0.25])
        )


class PhredErrorModel(AbstractErrorModel):
    """
    A simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicPhredScoreDistribution.
    """
    def __init__(self, insertion_error_ll: float, deletion_error_ll: float, read_len: int = 150):
        """
        :param read_len: DEPRECATED. Specifies the length of reads.
        """
        self.insertion_error_ll = insertion_error_ll
        self.deletion_error_ll = deletion_error_ll
        self.q_dist = BasicPhredScoreDistribution(length=read_len)  # These are deprecated/not being properly used.

    # noinspection PyUnusedLocal
    def indel_ll(self, read: SequenceRead, insertions: np.ndarray, deletions: np.ndarray):
        insertion_ll = np.sum(insertions) * (self.insertion_error_ll - np.log(4))
        deletion_ll = np.sum(deletions) * self.deletion_error_ll
        return insertion_ll + deletion_ll

    def compute_log_likelihood(self,
                               fragment: np.ndarray,
                               read: SequenceRead,
                               read_reverse_complemented: bool,
                               insertions: np.ndarray,
                               deletions: np.ndarray,
                               read_start_clip: int = 0,
                               read_end_clip: int = 0) -> float:
        """
        Uses phred scores to compute Pr(Read | Fragment, Quality).
        """
        if read_reverse_complemented:
            read_qual = read.quality[::-1]
            read_seq = read.seq.revcomp_bytes()
        else:
            read_qual = read.quality
            read_seq = read.seq.bytes()

        # take care of insertions/deletions/clipping
        # (NOTE: after reverse complement transformation of read, if necessary)
        _slice = slice(read_start_clip, len(read_seq) - read_end_clip)
        read_qual = read_qual[_slice][~insertions]
        read_seq = read_seq[_slice][~insertions]
        fragment_seq = fragment[~deletions]

        error_log10_prob = -0.1 * read_qual
        matches: np.ndarray = (fragment_seq == read_seq) & (read_qual > 0)
        mismatches: np.ndarray = (fragment_seq != read_seq) & (read_seq != bytes_N)

        """
        Phred model: Pr(measured base = 'A', true base = 'G' | q) = ( 1/3 * 10^{-q/10} )
        """
        log_p_errors = -np.log(3) + np.log(10) * error_log10_prob[np.where(mismatches)]
        log_p_matches = np.log(1 - np.power(10, error_log10_prob[np.where(matches)]))
        return self.indel_ll(read, insertions, deletions) + log_p_matches.sum() + log_p_errors.sum()
