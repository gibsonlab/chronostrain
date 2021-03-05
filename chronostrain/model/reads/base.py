"""
    Contains abstract classes. See the other python files for implementations.
"""
import torch
from abc import abstractmethod, ABCMeta
from chronostrain.model import Fragment


class SequenceRead:
    """
    A class representing a sequence-quality vector pair.
    """
    def __init__(self, seq: str, quality: torch.Tensor, metadata: str):
        if len(seq) != len(quality):
            raise ValueError(
                "Length of nucleotide sequence ({}) must agree with length of quality score sequence ({})".format(
                    len(seq), len(quality)
                )
            )
        self.seq = seq
        self.quality = quality
        self.metadata = metadata


class AbstractErrorModel(metaclass=ABCMeta):
    """
    Parent class for all fragment-to-read error models. This class (and its implementations) determines the
    likelihood values of the model (and therefore all inference algorithms).
    """

    @abstractmethod
    def compute_log_likelihood(self, fragment: Fragment, read: SequenceRead) -> float:
        """
        Compute the log probability of observing the read, conditional on the fragment.
        :param fragment: The source fragment (a String)
        :param read: The read (of type SequenceRead)
        :return: the value P(read | fragment).
        """
        pass

    @abstractmethod
    def sample_noisy_read(self, fragment: str, metadata: str = "") -> SequenceRead:
        """
        Obtain a random read (q-vec and sequence pair) from a given fragment.

        :param fragment: The source fragment.
        :param metadata: The metadata to store in the read.
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


class AbstractQScoreDistribution(metaclass=ABCMeta):
    """
    Parent class for all Q-score distributions. Meant as a utility function to help compute log likelihoods in
    error models (AbstractErrorModel).
    """

    def __init__(self, length):
        self.length = length

    @abstractmethod
    def compute_log_likelihood(self, qvec) -> float:
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
