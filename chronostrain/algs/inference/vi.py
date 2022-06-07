"""
vi.py
This is the second-order approximation solution for VI derived in a previous writeup.
(Note: doesn't work as well as mean-field ADVI.)
"""
from abc import ABCMeta, abstractmethod
from pathlib import Path

import torch
from torch.distributions import Normal


class AbstractPosterior(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Returns a sample from this posterior distribution.
        :param num_samples: the number of samples (N).
        :return: A time-indexed, simplex-valued (T x N x S) abundance tensor.
        """
        pass

    @abstractmethod
    def mean(self) -> torch.Tensor:
        """
        Returns the mean of this posterior distribution.
        :return: A time-indexed (T x S) abundance tensor.
        """
        pass

    @abstractmethod
    def log_likelihood(self, x: torch.Tensor) -> float:
        pass

    @abstractmethod
    def save(self, target_path: Path):
        pass
