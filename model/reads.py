"""
 reads.py
 Contains classes for the error model of reads.
"""
from abc import ABC, abstractmethod

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
    def sample_noisy_reads(self, fragment, num_samples=1):
        """
        Obtain a random sample (or multiple samples)
        :param fragment: The source fragment.
        :param num_samples: The number of iid samples to obtain conditional on the fragment (default 1).
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


class BasicErrorModel(AbstractErrorModel):
    def __init__(self, stuff):
        pass

    def compute_likelihood(self, fragment, read):
        pass

    def sample_noisy_reads(self, fragment, num_samples=1):
        pass