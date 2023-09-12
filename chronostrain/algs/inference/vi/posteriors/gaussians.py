from abc import ABC
from pathlib import Path
from typing import *

import jax
import jax.numpy as np
import numpy as cnp

from chronostrain.model.generative import GenerativeModel
from ..base import AbstractReparametrizedPosterior, _GENERIC_SAMPLE_TYPE, _GENERIC_PARAM_TYPE
from .util import tril_linear_transform_with_bias, gaussian_entropy

from chronostrain.config import cfg
from chronostrain.logging import create_logger
logger = create_logger(__name__)


INIT_SCALE = 1.0


class ReparametrizedGaussianPosterior(AbstractReparametrizedPosterior, ABC):
    def __init__(self, num_strains: int, num_times: int):
        self.num_strains = num_strains
        self.num_times = num_times

    def random_sample(self, num_samples: int) -> _GENERIC_SAMPLE_TYPE:
        return {
            'std_gaussians': jax.random.normal(
                shape=[num_samples, self.num_times * self.num_strains],
                key=next(cfg.engine_cfg.generate_prng_keys(num_keys=1))
            )
        }


class GaussianPosteriorFullReparametrizedCorrelation(ReparametrizedGaussianPosterior):

    def __init__(self, num_strains: int, num_times: int, dtype, initial_gaussian_bias: Optional[np.ndarray] = None):
        """
        Mean-field assumption:
        1) Parametrize the (T x S) trajectory as a (TS)-dimensional Gaussian.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        """
        logger.info("Initializing Fully joint posterior")
        super().__init__(num_strains, num_times)

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        n_features = self.num_times * self.num_strains
        self.parameters = {
            'tril_weights': np.zeros((n_features * (n_features - 1)) // 2, dtype=dtype),
            'diag_weights': np.full(n_features, fill_value=cnp.log(INIT_SCALE), dtype=dtype),
        }
        if initial_gaussian_bias is None:
            self.parameters['bias'] = np.zeros(n_features, dtype=dtype)
        else:
            self.parameters['bias'] = np.ravel(initial_gaussian_bias)

    def set_parameters(self, params: _GENERIC_PARAM_TYPE):
        self.parameters = params

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return self.parameters

    def reparametrize(self, random_samples: _GENERIC_SAMPLE_TYPE, params: _GENERIC_PARAM_TYPE) -> np.ndarray:
        n_samples = random_samples['std_gaussians'].shape[0]
        return tril_linear_transform_with_bias(
            params['tril_weights'],
            np.exp(params['diag_weights']),
            params['bias'],
            random_samples['std_gaussians']
        ).reshape(n_samples, self.num_times, self.num_strains).transpose([1, 0, 2])

    def abundance_sample(self, num_samples: int = 1) -> np.ndarray:
        x = self.reparametrize(self.random_sample(num_samples), self.get_parameters())
        return jax.nn.softmax(x, axis=-1)

    def entropy(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        return gaussian_entropy(
            params['tril_weights'], np.exp(params['diag_weights'])
        )


# class GaussianPosteriorStrainCorrelation(ReparametrizedGaussianPosterior):
#     def __init__(self, model: GenerativeModel):
#         """
#         Mean-field assumption:
#         1) Parametrize X_1, ..., X_T as independent S-dimensional gaussians.
#         2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
#         """
#         logger.info("Initializing Time-factorized (strain-correlated) posterior")
#         super().__init__(model)
#         self.num_strains = model.num_strains()
#         self.num_times = model.num_times()
#
#         # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
#         self.reparam_networks = []
#
#         for _ in range(self.num_times):
#             n_features = self.num_strains
#             linear_layer = TrilLinear(
#                 n_features=n_features,
#                 bias=True,
#                 device=cfg.engine_cfg.device
#             )
#             init_diag(linear_layer.weight, scale=np.log(INIT_SCALE))
#             torch.nn.init.constant_(linear_layer.bias, 0)
#             self.reparam_networks.append(linear_layer)
#
#         self.parameters = []
#         for network in self.reparam_networks:
#             self.parameters += network.parameters()
#
#         self.standard_normal = Normal(
#             loc=torch.tensor(0.0, device=cfg.engine_cfg.device),
#             scale=torch.tensor(1.0, device=cfg.engine_cfg.device)
#         )
#
#     def trainable_parameters(self) -> List[Parameter]:
#         return self.trainable_mean_parameters() + self.trainable_variance_parameters()
#
#     def trainable_mean_parameters(self) -> List[Parameter]:
#         return [m.bias for m in self.reparam_networks]
#
#     def trainable_variance_parameters(self) -> List[Parameter]:
#         # noinspection PyTypeChecker
#         return [m.weight for m in self.reparam_networks]
#
#     def mean(self) -> torch.Tensor:
#         return torch.stack([
#             self.reparam_networks[t].bias.detach()
#             for t in range(self.num_times)
#         ], dim=0)
#
#     def entropy(self) -> torch.Tensor:
#         parts = [
#             torch.distributions.MultivariateNormal(
#                 loc=net.bias,
#                 scale_tril=net.weight
#             ).entropy()
#             for net in self.reparam_networks
#         ]
#         return torch.sum(torch.stack(parts))
#
#     def differentiable_sample(self,
#                               num_samples=1,
#                               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         std_gaussian_samples = self.standard_normal.sample(
#             sample_shape=torch.Size((self.num_times, num_samples, self.num_strains))
#         )
#
#         # ======= Reparametrization
#         samples = torch.stack([
#             self.reparam_networks[t].forward(std_gaussian_samples[t, :, :])
#             for t in range(self.num_times)
#         ], dim=0)  # (T x N x S)
#
#         return samples
#
#     def log_likelihood(self, samples: torch.Tensor):
#         # input is (T x N x S)
#         n_samples = samples.size()[1]
#         ans = torch.zeros(size=(n_samples,), requires_grad=True, device=cfg.engine_cfg.device)
#         for t in range(self.num_times):
#             samples_t = samples[t]
#             linear = self.reparam_networks[t]
#             try:
#                 log_likelihood_t = torch.distributions.MultivariateNormal(
#                     loc=linear.bias,
#                     scale_tril=linear.weight
#                 ).log_prob(samples_t)
#             except ValueError:
#                 logger.error(f"Problem while computing log MV log-likelihood of time index {t}.")
#                 raise
#             ans = ans + log_likelihood_t
#         return ans
#
#     def save(self, path: Path):
#         params = {}
#         for t_idx in range(self.num_times):
#             linear_layer = self.reparam_networks[t_idx]
#             params[t_idx] = {
#                 "weight": linear_layer.weight.detach().cpu(),
#                 "bias": linear_layer.bias.detach().cpu()
#             }
#         torch.save(params, path)


# class GaussianPosteriorTimeCorrelation(ReparametrizedGaussianPosterior):
#     def __init__(self, model: GenerativeModel):
#         """
#         Mean-field assumption:
#         1) Parametrize X_1, X_2, ..., X_S as independent T-dimensional gaussians (one per strain).
#         2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
#         """
#         logger.info("Initializing Strain-factorized (time-correlated) posterior")
#         super().__init__(model)
#         self.num_times = model.num_times()
#         self.num_strains = model.num_strains()
#
#         # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
#         self.reparam_networks: List[torch.nn.Module] = []
#
#         for s_idx in range(self.num_strains):
#             n_features = self.num_times
#             linear_layer = TrilLinear(
#                 n_features=n_features,
#                 bias=True,
#                 device=cfg.engine_cfg.device
#             )
#             init_diag(linear_layer.weight, scale=np.log(INIT_SCALE))
#             torch.nn.init.constant_(linear_layer.bias, 0.0)
#             self.reparam_networks.append(linear_layer)
#         self.parameters = []
#         for network in self.reparam_networks:
#             self.parameters += network.parameters()
#
#         self.standard_normal = Normal(
#             loc=torch.tensor(0.0, device=cfg.engine_cfg.device),
#             scale=torch.tensor(1.0, device=cfg.engine_cfg.device)
#         )
#
#     def trainable_parameters(self) -> List[Parameter]:
#         return self.trainable_mean_parameters() + self.trainable_variance_parameters()
#
#     def trainable_mean_parameters(self) -> List[Parameter]:
#         return [m.bias for m in self.reparam_networks]
#
#     def trainable_variance_parameters(self) -> List[Parameter]:
#         # noinspection PyTypeChecker
#         return [m.weight for m in self.reparam_networks]
#
#     def mean(self) -> torch.Tensor:
#         return torch.stack([
#             net.bias.detach()
#             for net in self.reparam_networks
#         ], dim=1)
#
#     def entropy(self) -> torch.Tensor:
#         parts = [
#             torch.distributions.MultivariateNormal(
#                 loc=net.bias,
#                 scale_tril=net.weight
#             ).entropy()
#             for net in self.reparam_networks
#         ]
#         return torch.sum(torch.stack(parts))
#
#     def differentiable_sample(self,
#                               num_samples=1
#                               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         std_gaussian_samples = self.standard_normal.sample(
#             sample_shape=torch.Size((self.num_strains, num_samples, self.num_times))
#         )
#
#         # ======= Reparametrization
#         samples = torch.stack([
#             self.reparam_networks[s_idx].forward(std_gaussian_samples[s_idx, :, :])
#             for s_idx in range(self.num_strains)
#         ], dim=0)
#
#         return samples.transpose(0, 2)
#
#     def log_likelihood(self, samples: torch.Tensor):
#         n_samples = samples.size()[1]
#         ans = torch.zeros(size=(n_samples,), requires_grad=True, device=cfg.engine_cfg.device)
#         for s in range(self.num_strains):
#             samples_s = samples[:, :, s].t()
#             linear = self.reparam_networks[s]
#             w = linear.weight
#             try:
#                 log_likelihood_s = torch.distributions.MultivariateNormal(
#                     loc=linear.bias,
#                     scale_tril=w
#                 ).log_prob(samples_s)
#             except ValueError:
#                 logger.error(f"Problem while computing log MV log-likelihood of strain index {s}.")
#                 raise
#             ans = ans + log_likelihood_s
#         return ans
#
#     def save(self, path: Path):
#         params = {}
#         for s_idx in range(self.num_strains):
#             linear_layer = self.reparam_networks[s_idx]
#             params[s_idx] = {
#                 "weight": linear_layer.weight.detach().cpu(),
#                 "bias": linear_layer.bias.detach().cpu()
#             }
#         torch.save(params, path)
#
#     @staticmethod
#     def load(path: Path, model: GenerativeModel) -> 'GaussianPosteriorTimeCorrelation':
#         posterior = GaussianPosteriorTimeCorrelation(model)
#         params = torch.load(path)
#         for s_idx in range(model.num_strains()):
#             linear_layer = posterior.reparam_networks[s_idx]
#             linear_layer.weight = torch.nn.Parameter(params[s_idx]['weight'])
#             linear_layer.bias = torch.nn.Parameter(params[s_idx]['bias'])
#         return posterior
