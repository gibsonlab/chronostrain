"""
 reads.py
 Contains classes for the error model of reads.
"""
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np


class Q_Score(Enum):
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    TERRIBLE = 0


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
    def compute_likelihood(self, fragment, read):
        """
        Compute the probability of observing the read, conditional on the fragment.
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

    def __init__(self, length, distribution=np.array([0.05, 0.15, 0.3, 0.5])):
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

        lengths = self.length * np.array([distribution[Q_Score.TERRIBLE.value] / 2,
                                          distribution[Q_Score.LOW.value] / 2,
                                          distribution[Q_Score.MEDIUM.value] / 2,
                                          distribution[Q_Score.HIGH.value],
                                          distribution[Q_Score.MEDIUM.value] / 2,
                                          distribution[Q_Score.LOW.value] / 2,
                                          distribution[Q_Score.TERRIBLE.value] / 2])

        # Allocate a fixed-length array (we already know the length).
        quality_vector = np.zeros(self.length)

        # Hardcoded distribution
        cur = 0
        quality_vector[cur:cur+lengths[0]] = Q_Score.TERRIBLE.value

        cur = cur + lengths[0]
        quality_vector[cur:cur+lengths[1]] = Q_Score.LOW.value

        cur = cur + lengths[1]
        quality_vector[cur:cur + lengths[2]] = Q_Score.MEDIUM.value

        cur = cur + lengths[2]
        quality_vector[cur:cur + lengths[3]] = Q_Score.HIGH.value

        cur = cur + lengths[3]
        quality_vector[cur:cur + lengths[4]] = Q_Score.MEDIUM.value

        cur = cur + lengths[4]
        quality_vector[cur:cur + lengths[5]] = Q_Score.LOW.value

        cur = cur + lengths[5]
        quality_vector[cur:cur + lengths[6]] = Q_Score.TERRIBLE.value

        return quality_vector


class BasicErrorModel(AbstractErrorModel):
    """
    A very simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicQScoreDistribution.
    """

    # Define base change probability matrices conditioned on quality score level.

    # Example:
    # HIGH_Q_BASE_CHANGE_MATRIX[_A][_U] is the probability of observing U when the actual
    # nucleotide is _A

    base_indices = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    TERRIBLE_Q_BASE_CHANGE_MATRIX = np.array(([0.25, 0.25, 0.25, 0.25],
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

    Q_SCORE_BASE_CHANGE_MATRICES = [TERRIBLE_Q_BASE_CHANGE_MATRIX,
                                    LOW_Q_BASE_CHANGE_MATRIX,
                                    MEDIUM_Q_BASE_CHANGE_MATRIX,
                                    HIGH_Q_BASE_CHANGE_MATRIX]

    def __init__(self, read_len=150):
        self.read_len = read_len
        self.q_dist = BasicQScoreDistribution(read_len)

    def compute_likelihood(self, fragment, read):
        # TODO
        # TODO -- implement this
        # TODO
        pass

    def sample_noisy_read(self, fragment):
        quality_score_vector = self.q_dist.sample_qvec()

        noisy_fragment_chars = ['' for _ in range(self.read_len)]

        # Generate base pair reads from the sample fragment, conditioned on actual base pair and quality score for that base pair.
        for k, (actual_base_pair, q_score) in enumerate(zip(fragment, quality_score_vector)):
            actual_base_pair_index = BasicErrorModel.base_indices[actual_base_pair]

            # Generate a noisy base pair read from the distribution defined by the actual base pair and the quality score
            noisy_letter = np.random.choice(
                ["A", "C", "G", "T"],
                size=1,
                p=BasicErrorModel.Q_SCORE_BASE_CHANGE_MATRICES[q_score][actual_base_pair_index])[0]
            noisy_fragment_chars[k] = noisy_letter

        return ''.join(noisy_fragment_chars)
