from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional
from torch.distributions import Dirichlet, Normal, Uniform
import geotorch

from chronostrain.config import cfg, create_logger
from ..util import log_softmax
from .base import AbstractReparametrizedPosterior

logger = create_logger(__name__)


class BatchLinearTranspose(torch.nn.Module):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA + b`.
    Differs from base Linear module, which applies y = xA^T + b.
    Created to reinforce column-wise radial constraint.

    weight: the learnable weights of the module of shape (in_features) x (out_features).
    """

    def __init__(self, in_features: int, out_features: int, n_batches: int):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.empty(n_batches, out_features, in_features, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.weights.transpose(1, 2))


class ReparametrizedDirichletPosterior(AbstractReparametrizedPosterior):
    def __init__(self, num_strains: int, num_times: int):
        """
        Mean-field assumption:
        1) Parametrize X_1, ..., X_T as independent S-dimensional gaussians.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        """
        logger.info("Initializing Dirichlet posterior (S-1 dof)")
        self.num_strains = num_strains
        self.num_times = num_times

        self.log_concentrations = torch.nn.Parameter(
            np.log(0.5) + torch.zeros(
                self.num_times, self.num_strains,
                device=cfg.torch_cfg.device
            ),
            requires_grad=True
        )

        self.radial_network = BatchLinearTranspose(num_times, num_times, num_strains)
        geotorch.sphere(self.radial_network, "weights")
        # self.radial_network.weight = torch.nn.init.constant_(self.radial_network.weights, 1 / np.sqrt(num_times))
        with torch.no_grad():
            e = torch.stack([
                torch.eye(num_times, num_times, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
                for _ in range(num_strains)
            ], dim=0)
            self.radial_network.weights = self.radial_network.weights.copy_(e)

        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )
        self.standard_uniform = Uniform(
            low=torch.tensor(0.0, device=cfg.torch_cfg.device),
            high=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [self.log_concentrations] + list(self.radial_network.parameters())

    def mean(self) -> torch.Tensor:
        return torch.exp(
            self.log_concentrations - torch.logsumexp(self.log_concentrations, dim=1, keepdim=True)
        )

    def entropy(self) -> torch.Tensor:
        return Dirichlet(torch.exp(self.log_concentrations)).entropy().sum()

    def gaussian_approximation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Laplace gaussian-softmax reparametrized approximation, in the style of
        Srivasatava, Sutton: AUTOENCODING VARIATIONAL INFERENCE FOR TOPIC MODELS
        https://arxiv.org/pdf/1703.01488.pdf

        :return: The (T x S) mean and (T x S) standard deviation parameters.
        """
        mean = self.log_concentrations - torch.mean(self.log_concentrations, dim=1, keepdim=True)

        # Approximation uses diagonal covariance.
        inv_conc = torch.exp(-self.log_concentrations)
        scaling = torch.sqrt(
            (1 - (2 / self.num_strains)) * inv_conc
            + (1 / self.num_strains) * torch.mean(inv_conc, dim=1, keepdim=True)
        )
        return mean, scaling

    def reparametrized_sample_gaussian(self,
                                       num_samples=1
                                       ) -> torch.Tensor:
        """
        Return dirichlet samples (in log-simplex space) using the laplace gaussian-softmax approximation.
        :param num_samples:
        :return:
        """
        # std_gaussian_samples = self.standard_normal.sample(
        #     sample_shape=(self.num_times, num_samples, self.num_strains)
        # )
        # mean, scaling = self.gaussian_approximation()
        # return log_softmax(
        #     torch.unsqueeze(mean, 1) + torch.unsqueeze(scaling, 1) * std_gaussian_samples
        # )

        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(self.num_strains, num_samples, self.num_times)
        )
        mean, scaling = self.gaussian_approximation()

        # (S x N x T) @@ (S x T* x T) -> (S x N x T)   T*: radially normalized
        rotated = self.radial_network.forward(std_gaussian_samples).transpose(0, 2)
        return log_softmax(
            torch.unsqueeze(mean, 1) + torch.unsqueeze(scaling, 1) * rotated
        )

    def reparametrized_sample_icdf(self, num_samples: int) -> torch.Tensor:
        """
        Return dirichlet samples (in log-simplex space) using the approximate inverse-gamma CDF.
        :param num_samples:
        :return:
        """
        u = self.standard_uniform.sample(sample_shape=(self.num_times, num_samples, self.num_strains))
        log_alpha = self.log_concentrations.unsqueeze(1)
        alpha = torch.exp(log_alpha)
        log_diricihlet = torch.reciprocal(alpha) * (torch.log(u) + log_alpha + torch.lgamma(alpha))
        return log_diricihlet - torch.logsumexp(log_diricihlet, dim=-1, keepdim=True)

    def reparametrized_sample(self, num_samples: int) -> torch.Tensor:
        return self.reparametrized_sample_gaussian(num_samples)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        return torch.exp(self.reparametrized_sample(num_samples=num_samples).detach())

    def log_likelihood(self, log_dirichlet_samples: torch.Tensor):
        return Dirichlet(torch.exp(self.log_concentrations)).log_prob(torch.exp(log_dirichlet_samples).transpose(0, 1)).sum(dim=1)

    def save(self, path: Path):
        torch.save(self.log_concentrations.detach().cpu(), path)

    @staticmethod
    def load(path: Path, num_strains: int, num_times: int) -> 'ReparametrizedDirichletPosterior':
        posterior = ReparametrizedDirichletPosterior(num_strains, num_times)
        concs = torch.load(path)
        assert isinstance(concs, torch.Tensor)
        posterior.concentrations = concs
        return posterior
