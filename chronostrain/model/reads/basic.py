"""
    Classes written for simple toy examples, modelling deterministic Q scores
    and noisy reads (conditioned on these q-scores).
"""
import jax.numpy
import numpy as np
from chronostrain.model.reads.base import SequenceRead, AbstractErrorModel, AbstractQScoreDistribution


class RampUpRampDownDistribution(AbstractQScoreDistribution):
    """
        A default (simple) deterministic implementation, assigning probability 1 to a single particular quality score
        vector.

        Assigns a slow ramp-up  and then a ramp-down of the quality values over the target length.
        The outputted quality vector has quality assignments roughly proportional to the distribution specified.
    """

    def __init__(self,
                 length: int,
                 quality_score_values: np.array,
                 distribution: np.array):
        """
        :param length: the length of q-vectors to be generated.
        :param quality_score_values: the quality score values. Should be in increasing level of quality.
        :param distribution: An array of numbers, where the ith element describes the proportion for which
            the ith quality score should appear in the quality vector.
        """
        super().__init__()
        self.length = length

        if not (len(quality_score_values) == len(distribution)):
            raise ValueError("There must be exactly one ratio/frequency assigned for each quality score")

        self.distribution = distribution / distribution.sum()  # normalize to 1.
        self.quality_score_values = quality_score_values
        self.qvec = self.create_qvec()  # Hardcoded quality score vector.

    def compute_log_likelihood(self, qvec: np.ndarray):
        """
        Likelihood is just the indicator function.
        :param qvec: the query
        :return: 1 if query is qvec, 0 else.
        """
        return int(np.equal(qvec, self.qvec))

    def sample_qvec(self) -> np.ndarray:
        return self.qvec

    def create_qvec(self) -> np.ndarray:
        """
        Returns a single quality score vector ('q vector').


        The quality scores are chosen in proportion to the distribution specified
        in self.distribution.

        The current implementation assigns quality scores such that 1/2 of the specified
        frequency of the lowest score (quality_score_values[0]) is applied to the first base pairs in the fragment
        as well as the last base pairs in the fragment.

        This pattern is repeated with the next quality score (quality_score_values[1]) for the base pairs
        on both ends of the fragment that haven't been assigned a quality. The resulting quality
        pattern along the fragment is thus as follows:

            lowest QS region, medium QS region  ....     Highest QS region  ....     medium QS region,  lowest QS region

        :return: A list with length equal to the length of the input fragment list, and
        each element is a quality score from self.quality_score_values
        """

        lengths = np.zeros(shape=len(self.quality_score_values)*2-1, dtype=jax.numpy.uint16)

        # Iterate over each quality score and find the length of each chunk it will
        # span in the return vector.
        for index in range(len(self.quality_score_values)):
            if index == (len(self.quality_score_values)-1):  # Midpoint. Highest quality.
                lengths[index] = self.length * self.distribution[index]
            else:
                length = self.length * self.distribution[index] / 2
                lengths[index] = length
                lengths[len(lengths)-1-index] = length

        lengths = np.floor(lengths).astype(int)
        total_len = lengths.sum().item()

        if not (total_len <= self.length):
            raise ValueError("The sum of the segment lengths ({}) must be less than or equal "
                             "to the total fragment length ({})".format(total_len, self.length))

        # Allocate a fixed-length array.
        quality_vector = 10 * np.ones(self.length, dtype=int)

        # Note: The quality scores at the end of the vector do not get updated because of our flooring
        # in the lengths of above.

        # Go over the quality scores from lowest to highest, and then back down from highest to lowest.
        # assigning each quality score to its dedicated chunk of positions in the vector
        # according to the lengths of each chunk calculated previously.
        cur_pos = 0
        q_idx = 0
        incr = 1

        for i, count in enumerate(lengths):
            quality_vector[cur_pos:cur_pos + count] = self.quality_score_values[q_idx]
            cur_pos = cur_pos + count
            if q_idx == self.quality_score_values.shape[0] - 1:
                incr = -incr
            q_idx = q_idx + incr

        return quality_vector


class BasicQScoreDistribution(RampUpRampDownDistribution):

    def __init__(self, length: int):
        super().__init__(
            length,
            quality_score_values=np.array([0, 1, 2, 3, 4]),
            distribution=np.array([0.05, 0.15, 0.30, 0.25, 0.25])
        )


class BasicErrorModel(AbstractErrorModel):
    """
    A very simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicQScoreDistribution.
    """

    # Define base change probability matrices conditioned on quality score level.
    # Reminder: in Z4 representation, A/C/G/T = 0/1/2/3. The indices here are in the same order.

    # Example:
    # HIGH_Q_BASE_CHANGE_MATRIX[_A][_U] is the probability of observing U when the actual
    # nucleotide is _A. (e.g. each 1D array should sum to 1).

    VERYLOW_Q_BASE_CHANGE_MATRIX = [[0.25, 0.25, 0.25, 0.25],
                                    [0.25, 0.25, 0.25, 0.25],
                                    [0.25, 0.25, 0.25, 0.25],
                                    [0.25, 0.25, 0.25, 0.25]]

    LOW_Q_BASE_CHANGE_MATRIX = [[0.70, 0.10, 0.10, 0.10],
                                [0.10, 0.70, 0.10, 0.10],
                                [0.10, 0.10, 0.70, 0.10],
                                [0.10, 0.10, 0.10, 0.70]]

    MEDIUM_Q_BASE_CHANGE_MATRIX = [[0.85, 0.05, 0.05, 0.05],
                                   [0.05, 0.85, 0.05, 0.05],
                                   [0.05, 0.05, 0.85, 0.05],
                                   [0.05, 0.05, 0.05, 0.85]]

    HIGH_Q_BASE_CHANGE_MATRIX = [[0.91, 0.03, 0.03, 0.03],
                                 [0.03, 0.91, 0.03, 0.03],
                                 [0.03, 0.03, 0.91, 0.03],
                                 [0.03, 0.03, 0.03, 0.91]]

    VERYHIGH_Q_BASE_CHANGE_MATRIX = [[0.991, 0.003, 0.003, 0.003],
                                     [0.003, 0.991, 0.003, 0.003],
                                     [0.003, 0.003, 0.991, 0.003],
                                     [0.003, 0.003, 0.003, 0.991]]

    Q_SCORE_BASE_CHANGE_MATRICES = np.array([
        VERYLOW_Q_BASE_CHANGE_MATRIX,
        LOW_Q_BASE_CHANGE_MATRIX,
        MEDIUM_Q_BASE_CHANGE_MATRIX,
        HIGH_Q_BASE_CHANGE_MATRIX,
        VERYHIGH_Q_BASE_CHANGE_MATRIX
    ], dtype=float)

    def __init__(self, insertion_error_ll: float, deletion_error_ll: float, read_len: int = 150):
        self.read_len = read_len
        self.q_dist = BasicQScoreDistribution(read_len)
        self.insertion_error_ll = insertion_error_ll
        self.deletion_error_ll = deletion_error_ll

    def compute_log_likelihood(self,
                               fragment: np.ndarray,
                               read: SequenceRead,
                               read_reverse_complemented: bool,
                               insertions: np.ndarray,
                               deletions: np.ndarray,
                               read_start_clip: int = 0,
                               read_end_clip: int = 0) -> float:
        insertion_ll = np.sum(insertions) * self.insertion_error_ll
        deletion_ll = np.sum(deletions) * self.deletion_error_ll

        if read_reverse_complemented:
            read_qual = read.quality[::-1]
            read_seq = read.seq.revcomp_bytes()
        else:
            read_qual = read.quality
            read_seq = read.seq.bytes()

        # take care of insertions/deletions/clipping.
        _slice = slice(read_start_clip, len(read_seq) - read_end_clip)
        read_qual = read_qual[_slice][~insertions]
        read_seq = read_seq[_slice][~insertions]
        fragment_seq = fragment[~deletions]

        return insertion_ll + deletion_ll + np.log(
            BasicErrorModel.Q_SCORE_BASE_CHANGE_MATRICES[read_qual, fragment_seq, read_seq]
        ).sum()
