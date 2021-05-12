"""
vi.py
This is the second-order approximation solution for VI derived in a previous writeup.
(Note: doesn't work as well as mean-field BBVI.)
"""


from abc import ABCMeta, abstractmethod
import torch


class AbstractPosterior(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Returns a sample from this posterior distribution.
        :param num_samples: the number of samples (N).
        :return: A time-indexed (T x N x S) abundance tensor.
        """
        pass
