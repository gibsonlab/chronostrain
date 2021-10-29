from typing import Optional

import numpy as np
from chronostrain.model import Fragment
from chronostrain.model.reads.base import AbstractErrorModel, SequenceRead
from chronostrain.model.reads.basic import RampUpRampDownDistribution
from chronostrain.util.sequences import nucleotide_N_z4, NucleotideDtype


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

    def compute_log_likelihood(self,
                               fragment: Fragment,
                               read: SequenceRead,
                               read_reverse_complemented: bool,
                               insertions: Optional[np.ndarray] = None,
                               deletions: Optional[np.ndarray] = None) -> float:
        """
        Uses phred scores to compute Pr(Read | Fragment, Quality).
        """
        insertion_ll = np.sum(insertions) * self.insertion_error_ll
        deletion_ll = np.sum(deletions) * self.deletion_error_ll

        # take care of insertions.
        read_qual = read.quality[~insertions]
        read_seq = read.seq[~insertions]
        fragment_seq = fragment.seq[~deletions]

        if read_reverse_complemented:
            read_qual = read_qual[::-1]
            read_seq = read_seq[::-1]

        error_log10_prob = -0.1 * read_qual
        matches: np.ndarray = (fragment_seq == read_seq) & (read_qual > 0)
        mismatches: np.ndarray = (fragment_seq != read_seq) & (read_seq != nucleotide_N_z4)

        # TODO: N's might need to be included and handled differently.

        """
        Phred model: Pr(measured base = 'A', true base = 'G' | q) = ( 1/3 * 10^{-q/10} )
        """
        log_p_errors = -np.log(3) + np.log(10) * error_log10_prob[np.where(mismatches)]
        log_p_matches = np.log(1 - np.power(10, error_log10_prob[np.where(matches)]))
        return insertion_ll + deletion_ll + log_p_matches.sum() + log_p_errors.sum()

    def sample_noisy_read(self, read_id: str, fragment: Fragment, metadata="") -> SequenceRead:
        qvec = self.q_dist.sample_qvec()
        read = SequenceRead(read_id=read_id, seq=fragment.seq, quality=qvec, metadata=metadata)

        # Random shift by an integer mod 4.
        error_probs = np.power(10, -0.1 * qvec)
        error_locations: np.ndarray = np.less(np.random.rand(error_probs.shape[0]), error_probs)

        rand_shift = np.random.randint(low=0, high=4, size=np.sum(error_locations), dtype=NucleotideDtype)
        read.seq[np.where(error_locations)] = np.mod(
            read.seq[np.where(error_locations)] + rand_shift,
            4
        )
        return read
