"""
 reads.py
 Contains classes for the error model of reads.
"""
from abc import ABC, abstractmethod
import numpy as np
import math
import time

class SequenceRead:
    """
    A class representing a sequence-quality vector pair.
    """
    def __init__(self, seq: str, quality: list, metadata):
        if len(seq) != len(quality):
            raise ValueError("Length of nucleotide sequence ({}) must agree with length "
                             "of quality score sequence ({})".format(len(seq), len(quality))
                             )

        self.seq = seq
        self.quality = quality
        self.metadata = metadata


class AbstractQScoreDistribution(ABC):
    """
    Parent class for all Q-score distributions.
    """

    def __init__(self, length):
        self.length = length

    @abstractmethod
    def compute_likelihood(self, qvec) -> float:
        """
        Compute the likelihood of a given q-vector.
        :param qvec: The query.
        :return: the marginal probability P(qvec).
        """
        pass

    @abstractmethod
    def sample_qvec(self) -> list:
        """
        Obtain a random sample.
        :return: A quality score vector from the specified distribution.
        """
        pass


class AbstractErrorModel(ABC):
    """
    Parent class for all fragment-to-read error models.
    """

    @abstractmethod
    def compute_log_likelihood(self, fragment: str, read: SequenceRead):
        """
        Compute the log probability of observing the read, conditional on the fragment.
        :param fragment: The source fragment (a String)
        :param read: The read (of type SequenceRead)
        :return: the value P(read | fragment).
        """
        pass

    @abstractmethod
    def sample_noisy_read(self, fragment: str):
        """
        Obtain a random read (q-vec and sequence pair) from a given fragment.
        :param fragment: The source fragment.
        :return: A list of reads sampled according to their probabilities.
        """
        pass


class AbstractTrainableErrorModel(AbstractErrorModel):
    """
    Parent class of all trainable read error models.
    """

    @abstractmethod
    def train_from_data(self, reads, fragments, iters):
        """
        Attempt to train the error model from data.
        :param reads: A list of reads from a dataset.
        :param fragments: The corresponding collection of fragments.
        :param iters: the number of iterations.
        :return: None
        """
        pass


# ======================================================================================
# ==================================== IMPLEMENTATIONS =================================
# ======================================================================================


class RampUpRampDownDistribution(AbstractQScoreDistribution):
    """
        A default (simple) deterministic implementation, assigning probability 1 to a single particular quality score
        vector.

        Assigns a slow ramp-up  and then a ramp-down of the quality values over the target length.
        The outputted quality vector has quality assignments roughly proportional to the distribution specified.
    """

    def __init__(self,
                 length: int,
                 quality_score_values: np.ndarray,
                 distribution: np.ndarray):
        """
        :param length: the length of q-vectors to be generated.
        :param quality_score_values: the quality score values. Should be in increasing level of quality.
        :param distribution: An array of numbers, where the ith element describes the proportion for which
            the ith quality score should appear in the quality vector.
        """
        super().__init__(length)

        if not (len(quality_score_values) == len(distribution)):
            raise ValueError("There must be exactly one ratio/frequency assigned for each quality score")

        self.distribution = distribution / distribution.sum()  # normalize to 1.
        self.quality_score_values = quality_score_values
        self.qvec = self.create_qvec()  # Hardcoded quality score vector.

    def compute_likelihood(self, qvec):
        """
        Likelihood is just the indicator function.
        :param qvec: the query
        :return: 1 if query is qvec, 0 else.
        """
        return int(np.array_equal(qvec, self.qvec))

    def sample_qvec(self):
        return self.qvec

    def create_qvec(self):
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

        lengths = [0]*(len(self.quality_score_values)*2-1)

        # Iterate over each quality score and find the length of each chunk it will
        # span in the return vector.
        for index in range(len(self.quality_score_values)):
            if index == (len(self.quality_score_values)-1): # Midpoint. Highest quality.
                lengths[index] = self.length * self.distribution[index]
            else:
                length = self.length * self.distribution[index] / 2
                lengths[index] = length
                lengths[len(lengths)-1-index] = length

        lengths = [math.floor(i) for i in lengths]

        if not (np.cumsum(lengths)[-1] <= self.length):
            raise ValueError("The sum of the segment lengths ({}) must be less than or equal "
                             "to the total fragment length ({})".format(np.cumsum(lengths)[-1], self.length))

        # Allocate a fixed-length array to fill in with quality scores (default to quality score '10')
        quality_vector = np.ones(shape=self.length, dtype=int)*10

        # Note: The quality scores at the end of the vector
        # does not get updated because of our flooring in the lengths of above.

        # Go over the quality scores from lowest to highest, and then back down from highest to lowest.
        # assigning each quality score to its dedicated chunk of positions in the vector
        # according to the lengths of each chunk calculated previously.
        cur_pos = 0
        # for i, value in enumerate(np.append( self.quality_score_values, np.flip(self.quality_score_values[:-1]))):
        for i, value in enumerate(self.quality_score_values.tolist() + self.quality_score_values.tolist()[:-1][::-1]): ## TODO append
            if i > 0:
                cur_pos = cur_pos + lengths[i - 1]
            quality_vector[cur_pos:cur_pos + lengths[i]] = value

        return quality_vector


# ========================================================================
# ===== Basic implementation (0~5-scale quality score) ===================
# ========================================================================

class BasicQScoreDistribution(RampUpRampDownDistribution):

    def __init__(self, length):

        super().__init__(length,
                         quality_score_values=np.array([0, 1, 2, 3, 4]),
                         distribution=np.array([0.05, 0.15, 0.30, 0.25, 0.25]))


class BasicErrorModel(AbstractErrorModel):
    """
    A very simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicQScoreDistribution.
    """

    # Define base change probability matrices conditioned on quality score level.

    # Example:
    # HIGH_Q_BASE_CHANGE_MATRIX[_A][_U] is the probability of observing U when the actual
    # nucleotide is _A. (e.g. each 1D array should sum to 1).

    base_indices = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    VERYLOW_Q_BASE_CHANGE_MATRIX = np.array(([0.25, 0.25, 0.25, 0.25],
                                              [0.25, 0.25, 0.25, 0.25],
                                              [0.25, 0.25, 0.25, 0.25],
                                              [0.25, 0.25, 0.25, 0.25]))

    LOW_Q_BASE_CHANGE_MATRIX = np.array(([0.70, 0.10, 0.10, 0.10],
                                         [0.10, 0.70, 0.10, 0.10],
                                         [0.10, 0.10, 0.70, 0.10],
                                         [0.10, 0.10, 0.10, 0.70]))

    MEDIUM_Q_BASE_CHANGE_MATRIX = np.array(([0.85, 0.05, 0.05, 0.05],
                                            [0.05, 0.85, 0.05, 0.05],
                                            [0.05, 0.05, 0.85, 0.05],
                                            [0.05, 0.05, 0.05, 0.85]))

    HIGH_Q_BASE_CHANGE_MATRIX = np.array(([0.91, 0.03, 0.03, 0.03],
                                          [0.03, 0.91, 0.03, 0.03],
                                          [0.03, 0.03, 0.91, 0.03],
                                          [0.03, 0.03, 0.03, 0.91]))

    VERYHIGH_Q_BASE_CHANGE_MATRIX = np.array(([0.991, 0.003, 0.003, 0.003],
                                          [0.003, 0.991, 0.003, 0.003],
                                          [0.003, 0.003, 0.991, 0.003],
                                          [0.003, 0.003, 0.003, 0.991]))

    Q_SCORE_BASE_CHANGE_MATRICES = [VERYLOW_Q_BASE_CHANGE_MATRIX,
                                    LOW_Q_BASE_CHANGE_MATRIX,
                                    MEDIUM_Q_BASE_CHANGE_MATRIX,
                                    HIGH_Q_BASE_CHANGE_MATRIX,
                                    VERYHIGH_Q_BASE_CHANGE_MATRIX]

    def __init__(self, read_len=150):
        self.read_len = read_len
        self.q_dist = BasicQScoreDistribution(read_len)

    def compute_log_likelihood(self, fragment, read):
        """
        Computes the log likelihood of reading 'fragment' as 'read'

        @param -- read
            read is a SequenceRead object. It contains two vectors of equal length,
            one with the nucleotide sequence (a string of 'A', 'U', C' and 'G')
            as well as a quality vector (a numpy array of ints)

        @param -- fragment
            a string of 'A', 'T', 'C', and 'G'
        """

        log_product = 0
        for actual_base_pair, read_base_pair, q_score in zip(fragment, read.seq, read.quality):

            idx1 = BasicErrorModel.base_indices[actual_base_pair]
            idx2 = BasicErrorModel.base_indices[read_base_pair]

            # probability of observing 'read_base_pair' when the actual base pair is 'actual_base_pair'
            base_prob = BasicErrorModel.Q_SCORE_BASE_CHANGE_MATRICES[q_score][idx1][idx2]

            log_product += np.log(base_prob)

        return log_product

    def sample_noisy_read(self, fragment):
        quality_score_vector = self.q_dist.sample_qvec()

        noisy_fragment_chars = ['' for _ in range(self.read_len)]
        noisy_fragment_quality = np.zeros(shape=self.read_len, dtype=int)

        # Generate base pair reads from the fragment, conditioned on fragment base pairs and
        # the quality score for that base pair.
        for k, (read_base_pair, q_score) in enumerate(zip(fragment, quality_score_vector)):
            read_base_pair_index = BasicErrorModel.base_indices[read_base_pair]

            # Generate a noisy base pair reads from the distribution defined by the
            # actual fragment base pair and the quality score

            noisy_letter = np.random.choice(
                ["A", "C", "G", "T"],
                size=1,
                p=BasicErrorModel.Q_SCORE_BASE_CHANGE_MATRICES[q_score][read_base_pair_index])[0]

            noisy_fragment_chars[k] = noisy_letter
            noisy_fragment_quality[k] = q_score

        noisy_fragment_string = ''.join(noisy_fragment_chars)
        seq_read = SequenceRead(noisy_fragment_string, noisy_fragment_quality, "blank metadata")
        return seq_read


# ======================================================================
# =========== FastQ implementation =====================================
# ======================================================================

class BasicPhredScoreDistribution(RampUpRampDownDistribution):
    """
        An implementation of the quality score ramp-up ramp-down model, where the quality values are PHRED scores.
        ref: https://en.wikipedia.org/wiki/Phred_quality_score
    """

    def __init__(self, length):
        super().__init__(length,
                         quality_score_values=np.array([10, 20, 30, 40, 50]),
                         distribution=np.array([0.05, 0.15, 0.30, 0.25, 0.25]))


class PhredScoreDistribution(AbstractQScoreDistribution):
    # TODO: Implement something more sophisticated than a slow ramp up of and ramp down of quality
    # TODO: over the nucleotides. Monte Carlo simulations?
    pass


class FastQErrorModel(AbstractErrorModel):
    """
    A simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicPhredScoreDistribution.
    """

    def __init__(self, read_len=150):
        self.read_len = read_len
        self.q_dist = BasicPhredScoreDistribution(read_len)

    def compute_log_likelihood(self, fragment, read):
        """
        Computes the log likelihood of reading 'fragment' as 'read'

        @param -- read
            read is a SequenceRead object. It contains two vectors of equal length,
            one with the nucleotide sequence (a string of 'A', 'U', C' and 'G')
            as well as a quality vector (a numpy array of ints)

        @param -- fragment
            a string of 'A', 'U', 'C', and 'G'
        """


        # TODO: Compute in parallel for speedup?
        # ref: https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490

        log_product = 0
        log_likelihood_map = {}

        for fragment_base_pair, read_base_pair, q_score in zip(fragment.seq, read.seq, read.quality):
            # Get probability of observing 'read_base_pair' when the actual base pair is 'actual_base_pair'

            if (fragment_base_pair, read_base_pair, q_score) in log_likelihood_map:
                log_product += log_likelihood_map[(fragment_base_pair, read_base_pair, q_score)]
                continue

            accuracy = 1-np.power(10, -q_score/10)
            base_prob = accuracy if read_base_pair == fragment_base_pair else 1-accuracy
            log_likelihood = np.log(base_prob)

            log_likelihood_map[(fragment_base_pair, read_base_pair, q_score)] = log_likelihood

            log_product += log_likelihood

        return log_product

    def sample_noisy_read(self, fragment):

        quality_score_vector = self.q_dist.sample_qvec()

        noisy_fragment_chars = ['' for _ in range(self.read_len)]
        noisy_fragment_quality = np.zeros(shape=self.read_len, dtype=int)

        # Generate base pair reads from the fragment, conditioned on fragment base pairs and
        # the quality score for that base pair.
        for k, (read_base_pair, q_score) in enumerate(zip(fragment, quality_score_vector)):

            choice_list = [read_base_pair]
            for base in ["A", "C", "G", "T"]:
                if base != read_base_pair:
                    choice_list.append(base)

            accuracy = 1 - np.power(10, -q_score/10)
            # Generate a noisy base pair reads from the distribution defined by the
            # actual fragment base pair and the quality score
            noisy_letter = np.random.choice(
                choice_list,
                size=1,
                p=[accuracy, (1-accuracy)/3, (1-accuracy)/3, (1-accuracy)/3])[0]
            noisy_fragment_chars[k] = noisy_letter
            noisy_fragment_quality[k] = q_score

        noisy_fragment_string = ''.join(noisy_fragment_chars)
        seq_read = SequenceRead(noisy_fragment_string, noisy_fragment_quality, "blank metadata")

        return seq_read
