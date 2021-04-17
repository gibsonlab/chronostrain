from typing import Union
from numpy import pi
import torch


_pi = torch.tensor(pi)


class ScaleInverseChiSquared:
    def __init__(self, dof: Union[torch.Tensor, float], scale: Union[torch.Tensor, float]):
        if isinstance(dof, float):
            self.dof = torch.tensor(dof)
        else:
            self.dof = dof

        if isinstance(scale, float):
            self.scale = torch.tensor(scale)
        else:
            self.scale = scale

        alpha = 0.5 * self.dof
        beta = 0.5 * self.dof * self.scale
        self.underlying_gamma = torch.distributions.gamma.Gamma(concentration=alpha, rate=beta)

    def sample(self) -> torch.Tensor:
        return 1 / self.underlying_gamma.rsample().item()

    def log_constant(self) -> torch.Tensor:
        return ScaleInverseChiSquared.sics_log_constant(self.dof, self.scale)

    @staticmethod
    def sics_log_constant(dof: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Note: This was implemented separately from the non-static log_constant() method because of SICSGaussian's
        #   log_likelihood() method.
        half_dof = 0.5 * dof
        return half_dof * torch.log(scale * half_dof) - torch.lgamma(half_dof)


class SICSGaussian:
    def __init__(self, mean: torch.Tensor, dof: Union[torch.Tensor, float], scale: Union[torch.Tensor, float]):
        self.mu = mean
        self.gaussian_dim = mean.size()[-1]
        self.sics = ScaleInverseChiSquared(dof, scale)

    def log_likelihood(self, x: torch.Tensor, t: float = 1.0):
        """
        Compute the likelihood of a D-dimensional vector x, where x_i is Normal(mu_i, t*sigma^2), and sigma^2
        (the variance common to all x_i) is SICS(shape, scale).
        If x is an (N x D) tensor, outputs a length-N tensor of entry-wise log-likelihoods by interpreting
        each row of X as a sample.
        """
        # The formula is (2*pi*T)^{-N/2} *  C(dof, scale) / C(A, B), where
        #         1) C(dof, scale) is the normalizing constant for the SICS(dof, scale) distribution,
        #         2) A = dof + N
        #         3) B = (1 / A) * [ (dof * scale) + (x_1 - mu_1)^2 / T + ... + (x_N - mu_N)^2 / T ]

        if len(self.mu.size()) == 1 and len(x.size()) == 2:
            mu = torch.unsqueeze(self.mu, dim=0)
            gaussian_axis = 1
        elif len(self.mu.size()) == len(x.size()):
            mu = self.mu
            gaussian_axis = -1
        else:
            raise ValueError("Unrecognized input shape of x: {}".format(x.size()))

        a = self.sics.dof + self.gaussian_dim
        b = (1 / a) * (
                self.sics.dof * self.sics.scale
                + (1 / t) * torch.sum(torch.pow(x - mu, 2), dim=gaussian_axis)
        )

        return (
                -0.5 * self.gaussian_dim * torch.log(2 * _pi * t)
                + self.sics.log_constant()
                - ScaleInverseChiSquared.sics_log_constant(dof=a, scale=b)
        )
