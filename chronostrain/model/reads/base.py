"""
    Contains abstract classes. See the other python files for implementations.
"""
from typing import Union

import numpy as np
from abc import abstractmethod, ABCMeta
from chronostrain.model import Fragment
from chronostrain.util.sequences import nucleotides_to_z4, z4_to_nucleotides


class SequenceRead:
    """
    A class representing a sequence-quality vector pair.
    """
    def __init__(self, seq: Union[str, np.ndarray], quality: np.array, metadata: str):
        if len(seq) != len(quality):
            raise ValueError(
                "Length of nucleotide sequence ({}) must agree with length of quality score sequence ({})".format(
                    len(seq), len(quality)
                )
            )
        if type(seq) == str:
            self.seq: np.ndarray = nucleotides_to_z4(seq)
        elif type(seq) == np.ndarray:
            self.seq = seq
        self.quality: np.array = quality
        self.metadata: str = metadata

    def nucleotide_content(self) -> str:
        return z4_to_nucleotides(self.seq)

    def __str__(self):
        return "[SEQ:{},QUAL:{}]".format(
            z4_to_nucleotides(self.seq),
            self.quality.numpy()
        )

    def __repr__(self):
        return "[SEQ:{},QUAL:{}]".format(
            self.seq,
            self.quality.numpy()
        )

    def __len__(self):
        return self.seq.shape[0]


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
    def sample_noisy_read(self, fragment: Fragment, metadata: str = "") -> SequenceRead:
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
    def compute_log_likelihood(self, qvec: np.array) -> float:
        """
        Compute the likelihood of a given q-vector.
        :param qvec: The query.
        :return: the marginal probability P(qvec).
        """
        pass

    @abstractmethod
    def sample_qvec(self) -> np.array:
        """
        Obtain a random sample.
        :return: A quality score vector from the specified distribution.
        """
        pass
