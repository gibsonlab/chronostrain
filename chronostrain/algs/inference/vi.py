"""
vi.py
This is the second-order approximation solution for VI derived in a previous writeup.
(Note: doesn't work as well as mean-field BBVI.)
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
        :return: A time-indexed (T x N x S) abundance tensor.
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


class GaussianPosteriorFullCorrelation(AbstractPosterior):
    """
    A basic implementation, which specifies a bias vector `b` and linear transformation `A`.
    Samples using the reparametrization x = Az + b, where z is a vector of standard Gaussians, so that x has covariance
    (A.T)(A) and mean b.
    """

    def __init__(self, num_strains: int, num_times: int, bias: torch.Tensor, linear: torch.Tensor):
        if len(bias) != num_strains * num_times:
            raise ValueError(
                "Expected `mean` argument to be of length {} ({} strains x {} timepoints), but got {}.".format(
                    num_strains * num_times,
                    num_strains,
                    num_times,
                    len(bias)
                )
            )
        self.num_strains = num_strains
        self.num_times = num_times
        self.bias = bias
        self.linear = linear  # (M x TS)
        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=bias.device),
            scale=torch.tensor(1.0, device=bias.device)
        )

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        samples = self.standard_normal.sample(sample_shape=(num_samples, self.linear.shape[0]))  # N x M
        samples = samples @ self.linear + self.bias  # N x TS

        # Re-shape N x (TS) into (T x N x S).
        return torch.transpose(
            samples.reshape(num_samples, self.num_times, self.num_strains),
            0, 1
        )

    def mean(self) -> torch.Tensor:
        return self.bias

    def log_likelihood(self, x: torch.Tensor) -> float:
        raise NotImplementedError("Unable to compute log-likelihood for this model (considers singular cov matrices).")

    def save(self, target_path: Path):
        torch.save(
            {
                'bias': self.bias.detach().cpu(),
                'linear': self.linear.detach().cpu()
            },
            target_path
        )
