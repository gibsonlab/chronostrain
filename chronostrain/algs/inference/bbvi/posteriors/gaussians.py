from abc import ABC
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import Parameter
import torch.nn.functional

from .base import AbstractReparametrizedPosterior
from .util import TrilLinear, BatchedTrilLinear, init_diag

from chronostrain.config import cfg
from chronostrain.logging import create_logger
logger = create_logger(__name__)


INIT_SCALE = 1.0


class ReparametrizedGaussianPosterior(AbstractReparametrizedPosterior, ABC):
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        return torch.softmax(
            self.reparametrized_sample(num_samples=num_samples).detach(),
            dim=2
        )


class GaussianPosteriorFullReparametrizedCorrelation(ReparametrizedGaussianPosterior):
    def __init__(self, num_strains: int, num_times: int):
        """
        Mean-field assumption:
        1) Parametrize the (T x S) trajectory as a (TS)-dimensional Gaussian.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        """
        logger.info("Initializing Fully joint posterior")
        self.num_strains = num_strains
        self.num_times = num_times

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        n_features = self.num_times * self.num_strains
        self.reparam_network = TrilLinear(
            n_features=n_features,
            bias=True,
            device=cfg.torch_cfg.device
        )
        init_diag(self.reparam_network.weight, scale=np.log(INIT_SCALE))
        torch.nn.init.constant_(self.reparam_network.bias, 0)

        self.parameters = list(self.reparam_network.parameters())
        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def trainable_parameters(self) -> List[Parameter]:
        return self.trainable_mean_parameters() + self.trainable_variance_parameters()

    def trainable_mean_parameters(self) -> List[Parameter]:
        assert isinstance(self.reparam_network.bias, Parameter)
        return [self.reparam_network.bias]

    def trainable_variance_parameters(self) -> List[Parameter]:
        assert isinstance(self.reparam_network.weight, Parameter)
        return [self.reparam_network.weight]

    def mean(self) -> torch.Tensor:
        return self.reparam_network.bias.detach()

    def entropy(self) -> torch.Tensor:
        return torch.distributions.MultivariateNormal(
            loc=self.reparam_network.bias,
            scale_tril=self.reparam_network.cholesky_part
        ).entropy()

    def reparametrized_sample(self, num_samples=1) -> torch.Tensor:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(num_samples, self.num_times * self.num_strains)
        )

        """
        Reparametrization: x = a + bz, z ~ N(0,1).
        When computing log-likelihood p(x; a,b), it is important to keep a,b differentiable. e.g.
        output logp = f(a+bz; a, b) where f is the gaussian density N(a,b).
        """
        samples = self.reparam_network.forward(std_gaussian_samples)
        return samples.view(
            num_samples, self.num_times, self.num_strains
        ).transpose(0, 1)

    def log_likelihood(self, samples: torch.Tensor):
        num_samples = samples.shape[1]
        samples = samples.transpose(0, 1).view(num_samples, self.num_times * self.num_strains)
        try:
            return torch.distributions.MultivariateNormal(
                loc=self.reparam_network.bias,
                scale_tril=self.reparam_network.cholesky_part
            ).log_prob(samples)
        except ValueError:
            logger.error(f"Problem while computing log MV log-likelihood.")
            raise

    def save(self, path: Path):
        params = {
            "weight": self.reparam_network.weight.detach().cpu(),
            "bias": self.reparam_network.bias.detach().cpu()
        }
        torch.save(params, path)

    @staticmethod
    def load(path: Path, num_strains: int, num_times: int) -> 'GaussianPosteriorFullReparametrizedCorrelation':
        posterior = GaussianPosteriorFullReparametrizedCorrelation(num_strains, num_times)
        params = torch.load(path)
        posterior.reparam_network.weight = torch.nn.Parameter(params['weight'])
        posterior.reparam_network.bias = torch.nn.Parameter(params['bias'])
        return posterior


class GaussianPosteriorAutoregressiveReparametrizedCorrelation(ReparametrizedGaussianPosterior):
    def __init__(self, num_strains: int, num_times: int):
        """
        Mean-field assumption:
        1) Parametrize the (T x S) trajectory as a (TS)-dimensional Gaussian.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        """
        logger.info("Initializing Fully joint posterior (Autoregressive AR(1) Blocks)")
        self.num_strains = num_strains
        self.num_times = num_times

        self.bias = torch.nn.Parameter(torch.zeros(num_times, num_strains, device=cfg.torch_cfg.device), requires_grad=True)
        self.strain_corrs = BatchedTrilLinear(num_strains, num_strains, n_batches=num_times)
        self.time_corrs_sub = torch.nn.Parameter(torch.ones(num_strains, num_times - 1, device=cfg.torch_cfg.device), requires_grad=True)
        self.time_corrs_diag = torch.nn.Parameter(torch.zeros(num_strains, num_times, device=cfg.torch_cfg.device), requires_grad=True)
        with torch.no_grad():
            self.strain_corrs.weights.copy_(
                torch.zeros(num_strains, num_strains).expand(num_times, num_strains, num_strains).to(cfg.torch_cfg.device)
            )
        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

        self.test_reparametrization_equivalence()

    def trainable_parameters(self) -> List[Parameter]:
        return self.trainable_mean_parameters() + self.trainable_variance_parameters()

    def trainable_mean_parameters(self) -> List[Parameter]:
        return [self.bias]

    def trainable_variance_parameters(self) -> List[Parameter]:
        return [self.time_corrs_diag, self.time_corrs_sub] + list(self.strain_corrs.parameters())

    def mean(self) -> torch.Tensor:
        return self.bias.detach()

    def entropy(self) -> torch.Tensor:
        # note: diagonal entries are already in log-space.
        # The factor (1/2) in "det(Sigma)/2" is already included here, since these matrices are square roots.
        return self.strain_corrs.weights.diagonal(dim1=-2, dim2=-1).sum() + self.time_corrs_diag.sum()

    def test_reparametrization_equivalence(self, num_samples: int = 100, tol: float = 1e-6):
        with torch.no_grad():
            z = self.standard_normal.sample(
                sample_shape=(num_samples, self.num_times * self.num_strains)
            )
            diffs = torch.abs(self.reparam_slow(z) - self.reparam_fast(z))
            num_numerical_errors = torch.sum(diffs > tol)
            if num_numerical_errors > 0:
                logger.warning(f"Reparametrization equivalence check yielded {num_numerical_errors} discrepancies (Max Deviation = {torch.max(diffs)}).")

    def left_linear_transform(self) -> torch.Tensor:
        # Let `_A` be the left factor.
        time_corrs_block_diag = torch.block_diag(*(
                torch.diag_embed(self.time_corrs_diag.exp(), offset=0)
                + torch.diag_embed(self.time_corrs_sub, offset=-1)
        ))  # S x T x T -> ST x ST strided

        # rearrange the striding pattern
        strides_base = self.num_times * torch.arange(0, self.num_strains, 1, dtype=torch.long)
        ordering = torch.concat([
            strides_base + i
            for i in range(self.num_times)
        ])
        ord_x, ord_y = torch.meshgrid(ordering, ordering, indexing='ij')
        time_corrs_block_diag = time_corrs_block_diag[ord_x, ord_y]

        return torch.mm(time_corrs_block_diag, self.strain_corrs.cholesky_part)

    def reparam_slow(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: (N x (TS)) tensor.
        :return: Reparametrized (T x N x S) samples.
        """
        _A = self.left_linear_transform()

        return torch.reshape(
            torch.nn.functional.linear(z, _A),
            (z.shape[0], self.num_times, self.num_strains)
        ).transpose(0, 1) + self.bias.unsqueeze(1)

    def reparam_fast(self, z: torch.Tensor) -> torch.Tensor:
        """
        :param z: (N x (TS)) tensor.
        :return: Reparametrized (T x N x S) samples.
        """
        n = z.shape[0]
        x = self.strain_corrs.forward(z).reshape(n, self.num_times, self.num_strains)  # N x T x S

        time_corr = (
                torch.diag_embed(self.time_corrs_diag.exp(), offset=0)
                + torch.diag_embed(self.time_corrs_sub, offset=-1)
        )  # S x T x T

        # (S x N x T) @ (S x T x T) -> (S x N x T)
        x = torch.bmm(x.permute(2, 0, 1), time_corr.transpose(-1, -2))
        return x.transpose(0, 2) + self.bias.unsqueeze(1)

    def reparametrized_sample(self, num_samples=1) -> torch.Tensor:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(num_samples, self.num_times * self.num_strains)
        )
        return self.reparam_slow(std_gaussian_samples)

    def log_likelihood(self, samples: torch.Tensor):
        # Let `_A` be the left factor.
        _A = self.left_linear_transform()

        num_samples = samples.shape[1]
        samples = samples.transpose(0, 1).reshape(num_samples, self.num_times * self.num_strains)
        return torch.distributions.MultivariateNormal(
            loc=self.bias.reshape(self.num_times * self.num_strains),
            covariance_matrix=_A @ _A.transpose(0, 1)
        ).log_prob(samples)

    def save(self, path: Path):
        logger.warning("model-specific save() not yet implemented for {}.".format(self.__class__.__name__))

    @staticmethod
    def load(path: Path, num_strains: int, num_times: int) -> 'GaussianPosteriorBlockReparametrizedCorrelation':
        raise NotImplementedError()


class GaussianPosteriorStrainCorrelation(ReparametrizedGaussianPosterior):
    def __init__(self, num_strains: int, num_times: int):
        """
        Mean-field assumption:
        1) Parametrize X_1, ..., X_T as independent S-dimensional gaussians.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        """
        logger.info("Initializing Time-factorized (strain-correlated) posterior")
        self.num_strains = num_strains
        self.num_times = num_times

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_networks = []

        for _ in range(self.num_times):
            n_features = self.num_strains
            linear_layer = TrilLinear(
                n_features=n_features,
                bias=True,
                device=cfg.torch_cfg.device
            )
            init_diag(linear_layer.weight, scale=np.log(INIT_SCALE))
            torch.nn.init.constant_(linear_layer.bias, 0)
            self.reparam_networks.append(linear_layer)

        self.parameters = []
        for network in self.reparam_networks:
            self.parameters += network.parameters()

        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def trainable_parameters(self) -> List[Parameter]:
        return self.trainable_mean_parameters() + self.trainable_variance_parameters()

    def trainable_mean_parameters(self) -> List[Parameter]:
        return [m.bias for m in self.reparam_networks]

    def trainable_variance_parameters(self) -> List[Parameter]:
        # noinspection PyTypeChecker
        return [m.weight for m in self.reparam_networks]

    def mean(self) -> torch.Tensor:
        return torch.stack([
            self.reparam_networks[t].bias.detach()
            for t in range(self.num_times)
        ], dim=0)

    def entropy(self) -> torch.Tensor:
        parts = [
            torch.distributions.MultivariateNormal(
                loc=net.bias,
                scale_tril=net.cholesky_part
            ).entropy()
            for net in self.reparam_networks
        ]
        return torch.sum(torch.stack(parts))

    def reparametrized_sample(self,
                              num_samples=1,
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(self.num_times, num_samples, self.num_strains)
        )

        # ======= Reparametrization
        samples = torch.stack([
            self.reparam_networks[t].forward(std_gaussian_samples[t, :, :])
            for t in range(self.num_times)
        ], dim=0)  # (T x N x S)

        return samples

    def log_likelihood(self, samples: torch.Tensor):
        # input is (T x N x S)
        n_samples = samples.size()[1]
        ans = torch.zeros(size=(n_samples,), requires_grad=True, device=cfg.torch_cfg.device)
        for t in range(self.num_times):
            samples_t = samples[t]
            linear = self.reparam_networks[t]
            try:
                log_likelihood_t = torch.distributions.MultivariateNormal(
                    loc=linear.bias,
                    scale_tril=linear.cholesky_part
                ).log_prob(samples_t)
            except ValueError:
                logger.error(f"Problem while computing log MV log-likelihood of time index {t}.")
                raise
            ans = ans + log_likelihood_t
        return ans

    def save(self, path: Path):
        params = {}
        for t_idx in range(self.num_times):
            linear_layer = self.reparam_networks[t_idx]
            params[t_idx] = {
                "weight": linear_layer.weight.detach().cpu(),
                "bias": linear_layer.bias.detach().cpu()
            }
        torch.save(params, path)


class GaussianPosteriorTimeCorrelation(ReparametrizedGaussianPosterior):
    def __init__(self, num_strains: int, num_times: int):
        """
        Mean-field assumption:
        1) Parametrize X_1, X_2, ..., X_S as independent T-dimensional gaussians (one per strain).
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        """
        logger.info("Initializing Strain-factorized (time-correlated) posterior")
        self.num_times = num_times
        self.num_strains = num_strains

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_networks: List[torch.nn.Module] = []

        for s_idx in range(self.num_strains):
            n_features = self.num_times
            linear_layer = TrilLinear(
                n_features=n_features,
                bias=True,
                device=cfg.torch_cfg.device
            )
            init_diag(linear_layer.weight, scale=np.log(INIT_SCALE))
            torch.nn.init.constant_(linear_layer.bias, 0.0)
            self.reparam_networks.append(linear_layer)
        self.parameters = []
        for network in self.reparam_networks:
            self.parameters += network.parameters()

        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def trainable_parameters(self) -> List[Parameter]:
        return self.trainable_mean_parameters() + self.trainable_variance_parameters()

    def trainable_mean_parameters(self) -> List[Parameter]:
        return [m.bias for m in self.reparam_networks]

    def trainable_variance_parameters(self) -> List[Parameter]:
        # noinspection PyTypeChecker
        return [m.weight for m in self.reparam_networks]

    def mean(self) -> torch.Tensor:
        return torch.stack([
            net.bias.detach()
            for net in self.reparam_networks
        ], dim=1)

    def entropy(self) -> torch.Tensor:
        parts = [
            torch.distributions.MultivariateNormal(
                loc=net.bias,
                scale_tril=net.cholesky_part
            ).entropy()
            for net in self.reparam_networks
        ]
        return torch.sum(torch.stack(parts))

    def reparametrized_sample(self,
                              num_samples=1
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(self.num_strains, num_samples, self.num_times),
        )

        # ======= Reparametrization
        samples = torch.stack([
            self.reparam_networks[s_idx].forward(std_gaussian_samples[s_idx, :, :])
            for s_idx in range(self.num_strains)
        ], dim=0)

        return samples.transpose(0, 2)

    def log_likelihood(self, samples: torch.Tensor):
        n_samples = samples.size()[1]
        ans = torch.zeros(size=(n_samples,), requires_grad=True, device=cfg.torch_cfg.device)
        for s in range(self.num_strains):
            samples_s = samples[:, :, s].t()
            linear = self.reparam_networks[s]
            w = linear.cholesky_part
            try:
                log_likelihood_s = torch.distributions.MultivariateNormal(
                    loc=linear.bias,
                    scale_tril=w
                ).log_prob(samples_s)
            except ValueError:
                logger.error(f"Problem while computing log MV log-likelihood of strain index {s}.")
                raise
            ans = ans + log_likelihood_s
        return ans

    def save(self, path: Path):
        params = {}
        for s_idx in range(self.num_strains):
            linear_layer = self.reparam_networks[s_idx]
            params[s_idx] = {
                "weight": linear_layer.weight.detach().cpu(),
                "bias": linear_layer.bias.detach().cpu()
            }
        torch.save(params, path)

    @staticmethod
    def load(path: Path, num_strains: int, num_times: int) -> 'GaussianPosteriorTimeCorrelation':
        posterior = GaussianPosteriorTimeCorrelation(num_strains, num_times)
        params = torch.load(path)
        for s_idx in range(num_strains):
            linear_layer = posterior.reparam_networks[s_idx]
            linear_layer.weight = torch.nn.Parameter(params[s_idx]['weight'])
            linear_layer.bias = torch.nn.Parameter(params[s_idx]['bias'])
        return posterior
