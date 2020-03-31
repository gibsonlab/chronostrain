"""
 reads.py
 Contains classes for the error model of reads.
"""
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import model.generative as generative
import math

class Q_Score(Enum):
    VERYHIGH = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    VERYLOW = 0


class AbstractQScoreDistribution(ABC):
    """
    Parent class for all Q-score distributions.
    """

    def __init__(self, length):
        self.length = length

    @abstractmethod
    def compute_likelihood(self, qvec):
        """
        Compute the likelihood of a given q-vector.
        :param qvec: The query.
        :return: the marginal probability P(qvec).
        """
        pass

    @abstractmethod
    def sample_qvec(self):
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
    def compute_log_likelihood(self, fragment, read):
        """
        Compute the log probability of observing the read, conditional on the fragment.
        :param fragment: The source fragment (a String)
        :param read: The read (of type SequenceRead)
        :return: the value P(read | fragment).
        """
        pass

    @abstractmethod
    def sample_noisy_read(self, fragment):
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


class BasicQScoreDistribution(AbstractQScoreDistribution):
    """
    A default (simple) implementation. Assigns probability 1 to a single particular quality score vector.
    """

    def __init__(self, length, distribution=np.array([0.05, 0.15, 0.30, 0.25, 0.25])):
        """
        :param length: the length of q-vectors to be generated.
        :param distribution: An array of weights "terrible", "low", "medium" and "high" respectively.
        """
        super().__init__(length)
        self.distribution = distribution / distribution.sum()  # normalize to 1.
        self.qvec = self.create_qvec(self.distribution)  # Hardcoded quality score vector.

    def compute_likelihood(self, qvec):
        """
        Likelihood is just the indicator function.
        :param qvec: the query
        :return: 1 if query is qvec, 0 else.
        """
        return int(np.array_equal(qvec, self.qvec))

    def sample_qvec(self):
        return self.qvec

    def create_qvec(self, distribution):
        """
        Returns a single quality score vector ('q vector').


        The quality scores are chosen in proportion to the distribution specified
        in self.distribution.

        The current implementation assigns quality scores such that 1/2 of the specified
        frequency of the "terrible" score is applied to the first base pairs in the fragment
        as well as the last base pairs in the fragment.

        This pattern is repeated with the 'low quality' score with the base pairs
        on both ends of the fragment that haven't been assigned. The resulting quality
        pattern along the fragment is thus as follows:

            Terrible - Low - Medium - High - Medium - Low - Terrible

        :return: A list with length equal to the length of the input fragment list, and
        each element is an integer (0-3) representing a quality score.

        """

        lengths = self.length * np.array([distribution[Q_Score.VERYLOW.value] / 2,
                                          distribution[Q_Score.LOW.value] / 2,
                                          distribution[Q_Score.MEDIUM.value] / 2,
                                          distribution[Q_Score.HIGH.value] / 2,
                                          distribution[Q_Score.VERYHIGH.value],
                                          distribution[Q_Score.HIGH.value] / 2,
                                          distribution[Q_Score.MEDIUM.value] / 2,
                                          distribution[Q_Score.LOW.value] / 2,
                                          distribution[Q_Score.VERYLOW.value] / 2])

        lengths = [math.floor(i) for i in lengths]

        if not (np.cumsum(lengths)[-1] <= self.length):
            raise ValueError("The sum of the segment lengths ({}) must be less than or equal "
                             "to the total fragment length ({})".format(np.cumsum(lengths)[-1],  self.length))

        # Allocate a fixed-length array (we already know the length).
        quality_vector = np.zeros(self.length, dtype=int)

        # Hardcoded distribution
        # TODO: The quality scores at the end of the vector
        # TODO: do not get updated because of our flooring in the lengths of above. Fix somehow?
        cur = 0
        for i, value in enumerate([0, 1, 2, 3, 4, 3, 2, 1, 0]):
            if i > 0:
                cur = cur + lengths[i-1]
            quality_vector[cur:cur + lengths[i]] = value

        return quality_vector


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
            a string of 'A', 'U', 'C', and 'G'
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
        seq_read = generative.SequenceRead(noisy_fragment_string, noisy_fragment_quality, "blank metadata")
        return seq_read


class PhredScoreDistribution(AbstractQScoreDistribution):
    def __init__(self, length, distribution=np.array([1, 2, 3, 4, 2])):
        """
        :param length: the length of q-vectors to be generated.
        :param distribution: An array of weights "terrible", "low", "medium" and "high" respectively.
        """
        super().__init__(length)
        self.distribution = distribution / distribution.sum()  # normalize to 1.
        self.qvec = self.create_qvec(self.distribution)  # Hardcoded quality score vector.
        # TODO: Replace with hardcoded create_qvec with a monte carlo inspired version.

    def compute_likelihood(self, qvec):
        """
        Likelihood is just the indicator function.
        :param qvec: the query
        :return: 1 if query is qvec, 0 else.
        """
        return int(np.array_equal(qvec, self.qvec))

    def sample_qvec(self):
        return self.qvec

    def create_qvec(self, distribution):
        """
        Returns a single quality score vector ('q vector').


        The quality scores are chosen in proportion to the distribution specified
        in self.distribution.

        The current implementation assigns quality scores such that 1/2 of the specified
        frequency of the "terrible" score is applied to the first base pairs in the fragment
        as well as the last base pairs in the fragment.

        This pattern is repeated with the 'low quality' score with the base pairs
        on both ends of the fragment that haven't been assigned. The resulting quality
        pattern along the fragment is thus as follows:

            Terrible - Low - Medium - High - Medium - Low - Terrible

        :return: A list with length equal to the length of the input fragment list, and
        each element is an integer (0-3) representing a quality score.

        """

        lengths = self.length * np.array([distribution[Q_Score.VERYLOW.value] / 2,
                                          distribution[Q_Score.LOW.value] / 2,
                                          distribution[Q_Score.MEDIUM.value] / 2,
                                          distribution[Q_Score.HIGH.value] / 2,
                                          distribution[Q_Score.VERYHIGH.value],
                                          distribution[Q_Score.HIGH.value] / 2,
                                          distribution[Q_Score.MEDIUM.value] / 2,
                                          distribution[Q_Score.LOW.value] / 2,
                                          distribution[Q_Score.VERYLOW.value] / 2])

        lengths = [math.floor(i) for i in lengths]

        if not (np.cumsum(lengths)[-1] <= self.length):
            raise ValueError("The sum of the segment lengths ({}) must be less than or equal "
                             "to the total fragment length ({})".format(np.cumsum(lengths)[-1], self.length))

        # Allocate a fixed-length array (we already know the length).
        quality_vector = np.empty(self.length, dtype=int)
        quality_vector.fill(10)

        # Hardcoded distribution
        # TODO: The quality scores at the end of the vector
        # TODO: do not get updated because of our flooring in the lengths of above. Fix somehow?
        cur = 0
        for i, value in enumerate([10, 20, 30, 40, 50, 40, 30, 20, 10]):
            if i > 0:
                cur = cur + lengths[i-1]
            quality_vector[cur:cur + lengths[i]] = value

        return quality_vector


class FastQErrorModel(AbstractErrorModel):
    """
    A very simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicQScoreDistribution.
    """

    def __init__(self, read_len=150):
        self.read_len = read_len
        self.q_dist = PhredScoreDistribution(read_len)

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

        log_product = 0
        for actual_base_pair, read_base_pair, q_score in zip(fragment, read.seq, read.quality):

            # probability of observing 'read_base_pair' when the actual base pair is 'actual_base_pair'
            accuracy = 1-np.power(10, -q_score/10)

            base_prob = accuracy if read_base_pair == actual_base_pair else 1-accuracy

            log_product += np.log(base_prob)

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
        seq_read = generative.SequenceRead(noisy_fragment_string, noisy_fragment_quality, "blank metadata")

        return seq_read




