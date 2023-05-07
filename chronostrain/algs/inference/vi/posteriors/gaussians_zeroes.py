from abc import ABC
from pathlib import Path
from typing import List, Tuple, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Parameter
import torch.nn.functional

from chronostrain.model.generative import GenerativeModel
from chronostrain.model.zeros.gumbel import PopulationLocalZeros, PopulationGlobalZeros
from ..base import AbstractReparametrizedPosterior
from .util import TrilLinear, init_diag

from chronostrain.config import cfg
from chronostrain.logging import create_logger
logger = create_logger(__name__)


INIT_SCALE = 1.0


class ReparametrizedGaussianWithZerosPosterior(AbstractReparametrizedPosterior, ABC):
    def __init__(self, model: GenerativeModel):
        self.model = model

    def sample_gaussians(self, num_samples: int) -> Tensor:
        raise NotImplementedError()

    def sample(self, num_samples: int = 1) -> Tensor:
        return self.model.latent_conversion(
            self.sample_gaussians(num_samples=num_samples).detach()
            +
            torch.log(self.sample_zeroes(num_samples))
        )

    def sample_zeroes(self, num_samples: int) -> Tensor:
        raise NotImplementedError()


class GaussianWithGlobalZerosPosterior(ReparametrizedGaussianWithZerosPosterior):
    """
    A zero-model posterior, where there is a posterior indicator for each strain (to be applied across all timepoints).
    """
    def __init__(self, model: GenerativeModel, zero_model: PopulationGlobalZeros, inv_temp: float):
        logger.info("Initializing Fully joint posterior with Global Zeros")
        super().__init__(model)
        self.zero_model = zero_model
        self.num_strains = model.num_strains()
        self.num_times = model.num_times()

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        n_features = self.num_times * self.num_strains
        self.reparam_network = TrilLinear(
            n_features=n_features,
            bias=True,
            device=cfg.engine_cfg.device
        )
        init_diag(self.reparam_network.weight, scale=np.log(INIT_SCALE))
        torch.nn.init.constant_(self.reparam_network.bias, 0)

        # ========== Nonstandard Gumbel means
        self.gumbel_means = torch.nn.parameter.Parameter(
            torch.zeros(
                2, self.num_strains,
                device=cfg.engine_cfg.device
            )
        )

        # ========== Utility
        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.engine_cfg.device),
            scale=torch.tensor(1.0, device=cfg.engine_cfg.device)
        )
        self.standard_gumbel = torch.distributions.gumbel.Gumbel(
            loc=torch.tensor(0.0, device=cfg.engine_cfg.device),
            scale=torch.tensor(1.0, device=cfg.engine_cfg.device)
        )

        self.inv_temp = inv_temp

    def trainable_parameters(self) -> List[Parameter]:
        return list(self.reparam_network.parameters()) + [self.gumbel_means]

    def sample_zeroes(self, num_samples: int) -> Tensor:
        g = self.standard_gumbel.sample(
            torch.Size((2, num_samples, self.num_strains))
        ) + torch.unsqueeze(self.gumbel_means, dim=1)
        return torch.gt(g[0], g[1])

    def sample_smooth_log_zeroes(self, num_samples: int) -> Tensor:
        g = self.standard_gumbel.sample(
            torch.Size((2, num_samples, self.num_strains))
        ) + torch.unsqueeze(self.gumbel_means, dim=1)
        torch.nn.LogSigmoid()
        return torch.log_softmax(self.inv_temp * g, dim=0)[0]

    def sample_gaussians(self, num_samples: int) -> Tensor:
        """
        Gaussian Reparametrization: x = a + bz, z ~ N(0,1).
        See comments for GaussianReparametrizedPosterior.
        """
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=torch.Size((num_samples, self.num_times * self.num_strains))
        )
        return self.reparam_network.forward(std_gaussian_samples).view(
            num_samples, self.num_times, self.num_strains
        ).transpose(0, 1)

    def mean(self) -> Tensor:
        raise NotImplementedError()

    def entropy(self) -> Tensor:
        p = torch.softmax(self.gumbel_means, dim=0) # 2 x S, [p, 1-p]
        logp = torch.log_softmax(self.gumbel_means, dim=0) # 2 x S, [log(p), log(1-p)]
        binary_entropy = -torch.sum(p * logp)

        return torch.distributions.MultivariateNormal(
            loc=self.reparam_network.bias,
            scale_tril=self.reparam_network.weight
        ).entropy() + binary_entropy

    def differentiable_sample(self, num_samples: int) -> Tuple[Tensor, Tensor]:
        return self.sample_gaussians(num_samples), self.sample_smooth_log_zeroes(num_samples)

    def save(self, path: Path):
        params = {
            "weight": self.reparam_network.weight.detach().cpu(),
            "bias": self.reparam_network.bias.detach().cpu(),
            "gumbel_mean": self.gumbel_means
        }
        torch.save(params, path)


class GaussianWithGlobalZerosPosteriorSparsified(ReparametrizedGaussianWithZerosPosterior):
    """
    A zero-model posterior, where there is a posterior indicator for each strain (to be applied across all timepoints).
    """
    def __init__(self, model: GenerativeModel, zero_model: PopulationGlobalZeros, inv_temp: float):
        logger.info("Initializing Fully joint posterior with Global Zeros")
        super().__init__(model)
        self.zero_model = zero_model
        self.num_strains = model.num_strains()
        self.num_times = model.num_times()

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_networks = [
            TrilLinear(n_features=self.num_strains, bias=True, device=cfg.engine_cfg.device)
            for _ in range(self.num_times)
        ]
        self.cond_networks = [
            TrilLinear(n_features=self.num_strains, bias=False, device=cfg.engine_cfg.device)
            for _ in range(self.num_times - 1)
        ]

        # ========== Nonstandard Gumbel means
        self.gumbel_means = torch.nn.parameter.Parameter(
            torch.zeros(
                2, self.num_strains,
                device=cfg.engine_cfg.device
            )
        )

        # ========== Utility
        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.engine_cfg.device),
            scale=torch.tensor(1.0, device=cfg.engine_cfg.device)
        )
        self.standard_gumbel = torch.distributions.gumbel.Gumbel(
            loc=torch.tensor(0.0, device=cfg.engine_cfg.device),
            scale=torch.tensor(1.0, device=cfg.engine_cfg.device)
        )

        self.inv_temp = inv_temp

    def trainable_parameters(self) -> List[Parameter]:
        return list(self._all_parameters())

    def _all_parameters(self) -> Iterator[torch.nn.Parameter]:
        for n in self.reparam_networks:
            yield from n.parameters()
        for n in self.cond_networks:
            yield from n.parameters()
        yield self.gumbel_means

    def sample_zeroes(self, num_samples: int) -> Tensor:
        g = self.standard_gumbel.sample(
            torch.Size((2, num_samples, self.num_strains))
        ) + torch.unsqueeze(self.gumbel_means, dim=1)
        return torch.gt(g[0], g[1])

    def sample_smooth_log_zeroes(self, num_samples: int) -> Tensor:
        g = self.standard_gumbel.sample(
            torch.Size((2, num_samples, self.num_strains))
        ) + torch.unsqueeze(self.gumbel_means, dim=1)
        torch.nn.LogSigmoid()
        return torch.log_softmax(self.inv_temp * g, dim=0)[0]

    def sample_gaussians(self, num_samples: int) -> Tensor:
        """
        Gaussian Reparametrization: x = a + bz, z ~ N(0,1).
        See comments for GaussianReparametrizedPosterior.
        """
        gaussians = []
        for t_idx in range(self.num_times):
            z = self.standard_normal.sample(torch.Size((num_samples, self.num_strains)))
            if t_idx == 0:
                gaussians.append(
                    self.reparam_networks[0].forward(z)
                )
            else:
                gaussians.append(
                    self.reparam_networks[t_idx].forward(z) + self.cond_networks[t_idx-1].forward(gaussians[t_idx-1])
                )
        return torch.stack(gaussians, dim=0)

        # std_gaussian_samples = self.standard_normal.sample(
        #     sample_shape=torch.Size((num_samples, self.num_times * self.num_strains))
        # )
        # return self.reparam_network.forward(std_gaussian_samples).view(
        #     num_samples, self.num_times, self.num_strains
        # ).transpose(0, 1)

    def mean(self) -> Tensor:
        raise NotImplementedError()

    def entropy(self) -> Tensor:
        p = torch.softmax(self.gumbel_means, dim=0) # 2 x S, [p, 1-p]
        logp = torch.log_softmax(self.gumbel_means, dim=0)  # 2 x S, [log(p), log(1-p)]
        binary_entropy = -torch.sum(p * logp)

        entropies = torch.tensor([
            torch.distributions.MultivariateNormal(
                loc=net.bias,
                scale_tril=net.weight
            ).entropy()
            for net in self.reparam_networks
        ])

        return torch.sum(entropies) + binary_entropy

    def differentiable_sample(self, num_samples: int) -> Tuple[Tensor, Tensor]:
        return self.sample_gaussians(num_samples), self.sample_smooth_log_zeroes(num_samples)

    def save(self, path: Path):
        params = {
            "weights": [net.weight.detach().cpu() for net in self.reparam_networks],
            "biases": [net.bias.detach().cpu() for net in self.reparam_networks],
            "cond_weights": [net.weight.detach().cpu() for net in self.cond_networks],
            "gumbel_mean": self.gumbel_means
        }
        torch.save(params, path)


class GaussianWithLocalZeros(ReparametrizedGaussianWithZerosPosterior):
    def __init__(self, model: GenerativeModel, zero_model: PopulationLocalZeros):
        """
        Mean-field assumption:
        1) Parametrize the (T x S) trajectory as a (TS)-dimensional Gaussian.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        """
        logger.info("Initializing Fully joint posterior with Localized Zeros")
        super().__init__(model)
        self.zero_model = zero_model
        self.num_strains = model.num_strains()
        self.num_times = model.num_times()

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        n_features = self.num_times * self.num_strains
        self.reparam_network = TrilLinear(
            n_features=n_features,
            bias=True,
            device=cfg.engine_cfg.device
        )
        init_diag(self.reparam_network.weight, scale=np.log(INIT_SCALE))
        torch.nn.init.constant_(self.reparam_network.bias, 0)

        # ========== Nonstandard Gumbel means
        self.gumbel_means = torch.nn.parameter.Parameter(
            torch.zeros(
                2, self.num_times, self.num_strains,
                device=cfg.engine_cfg.device
            )
        )
        self.gumbel_between_means = torch.nn.parameter.Parameter(
            torch.zeros(
                2, self.num_times - 1, self.num_strains,
                device=cfg.engine_cfg.device
            )
        )

        # ========== Utility
        self.parameters = list(self.reparam_network.parameters()) + [self.gumbel_means, self.gumbel_between_means]
        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.engine_cfg.device),
            scale=torch.tensor(1.0, device=cfg.engine_cfg.device)
        )
        self.standard_gumbel = torch.distributions.gumbel.Gumbel(
            loc=torch.tensor(0.0, device=cfg.engine_cfg.device),
            scale=torch.tensor(1.0, device=cfg.engine_cfg.device)
        )

    def sample_gumbels(self, num_samples: int) -> Tuple[Tensor, Tensor]:
        gumbel_samples = self.standard_gumbel.sample(
            sample_shape=torch.Size((2, self.num_times, num_samples, self.num_strains,))
        ) + torch.unsqueeze(self.gumbel_means, dim=2)
        gumbel_between_samples = self.standard_gumbel.sample(
            sample_shape=torch.Size((2, (self.num_times - 1), num_samples, self.num_strains,))
        ) + torch.unsqueeze(self.gumbel_between_means, dim=2)
        return gumbel_samples, gumbel_between_samples

    def sample_gaussians(self, num_samples: int) -> Tensor:
        """
        Gaussian Reparametrization: x = a + bz, z ~ N(0,1).
        See comments for GaussianReparametrizedPosterior.
        """
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=torch.Size((num_samples, self.num_times * self.num_strains))
        )
        return self.reparam_network.forward(std_gaussian_samples).view(
            num_samples, self.num_times, self.num_strains
        ).transpose(0, 1)

    def sample_zeroes(self, num_samples: int) -> Tensor:
        gumbel_samples, gumbel_between_samples = self.sample_gumbels(num_samples)
        return self.zero_model.zeroes_of_gumbels(gumbel_samples, gumbel_between_samples)

    def trainable_parameters(self) -> List[Parameter]:
        return self.parameters

    def mean(self) -> Tensor:
        raise NotImplementedError()

    def entropy(self) -> Tensor:
        # Gumbels are parametrized by mean and thus only contributes a constant factor to entropy.
        return torch.distributions.MultivariateNormal(
            loc=self.reparam_network.bias,
            scale_tril=self.reparam_network.weight
        ).entropy()

    def differentiable_sample(self, num_samples=1) -> Tuple[Tensor, Tensor, Tensor]:
        gumbel_samples, gumbel_between_samples = self.sample_gumbels(num_samples)
        return self.sample_gaussians(num_samples), gumbel_samples, gumbel_between_samples

    def log_likelihood(self, samples: Tensor):
        raise NotImplementedError()

    def save(self, path: Path):
        params = {
            "weight": self.reparam_network.weight.detach().cpu(),
            "bias": self.reparam_network.bias.detach().cpu(),
            "gumbel_mean": self.gumbel_means,
            "gumbel_between_mean": self.gumbel_between_means
        }
        torch.save(params, path)

    @staticmethod
    def load(path: Path, model: GenerativeModel, zero_model: PopulationLocalZeros) -> 'GaussianWithLocalZeros':
        posterior = GaussianWithLocalZeros(model, zero_model)
        params = torch.load(path)
        posterior.reparam_network.weight = torch.nn.Parameter(params['weight'])
        posterior.reparam_network.bias = torch.nn.Parameter(params['bias'])
        posterior.gumbel_means = torch.nn.Parameter(params['gumbel_mean'])
        posterior.gumbel_between_means = torch.nn.Parameter(params['gumbel_between_mean'])
        return posterior
