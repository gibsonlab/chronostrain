from typing import Union, Tuple, Dict, List

import torch
from torch.distributions import Normal
from torch.nn import Parameter

from chronostrain.model.generative import GenerativeModel
from .base import AbstractReparametrizedPosterior
from .util import TrilLinear, init_diag

from chronostrain.config import cfg, create_logger
logger = create_logger(__name__)


class GaussianPosteriorFullCorrelation(AbstractReparametrizedPosterior):
    def __init__(self, model: GenerativeModel):
        """
        Mean-field assumption:
        1) Parametrize the (T x S) trajectory as a (TS)-dimensional Gaussian.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        :param model: The generative model to use.
        """
        super().__init__()

        # Check: might need this to be a matrix, not a vector.
        self.model = model

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_network = TrilLinear(
            n_features=self.model.num_times() * self.model.num_strains(),
            bias=True,
            device=cfg.torch_cfg.device
        )
        # torch.nn.init.eye_(self.reparam_network.weight)
        init_diag(self.reparam_network.weight, scale=1.0)
        self.parameters = self.reparam_network.parameters()
        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def trainable_parameters(self) -> List[Parameter]:
        return self.parameters

    def mean(self) -> torch.Tensor:
        return self.reparam_network.bias.detach()

    def reparametrized_sample(self, num_samples=1) -> torch.Tensor:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(num_samples, self.model.num_times() * self.model.num_strains())
        )

        """
        Reparametrization: x = a + bz, z ~ N(0,1).
        When computing log-likelihood p(x; a,b), it is important to keep a,b differentiable. e.g.
        output logp = f(a+bz; a, b) where f is the gaussian density N(a,b).
        """
        samples = self.reparam_network.forward(std_gaussian_samples)
        return samples.view(
            num_samples, self.model.num_times(), self.model.num_strains()
        ).transpose(0, 1)

    def reparametrized_sample_log_likelihoods(self, samples: torch.Tensor):
        w = self.reparam_network.cholesky_part
        try:
            return torch.distributions.MultivariateNormal(
                loc=self.reparam_network.bias,
                scale_tril=w
            ).log_prob(samples)
        except ValueError:
            logger.error(f"Problem while computing log MV log-likelihood.")
            raise

    def log_likelihood(self, samples: torch.Tensor) -> float:
        if len(samples.size()) == 2:
            r, c = samples.size()
            samples = samples.view(r, 1, c)
        return super().log_likelihood(samples)


class GaussianPosteriorStrainCorrelation(AbstractReparametrizedPosterior):
    def __init__(self, model: GenerativeModel):
        """
        Mean-field assumption:
        1) Parametrize X_1, ..., X_T as independent S-dimensional gaussians.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        :param model: The generative model to use.
        """
        # Check: might need this to be a matrix, not a vector.
        self.model = model

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_networks = []

        for _ in range(self.model.num_times()):
            linear_layer = TrilLinear(
                n_features=self.model.num_strains(),
                bias=True,
                device=cfg.torch_cfg.device
            )
            init_diag(linear_layer.weight, scale=1.0)
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
        return self.parameters

    def mean(self) -> torch.Tensor:
        return torch.stack([
            self.reparam_networks[t].bias.detach()
            for t in range(self.model.num_times())
        ], dim=0)

    def reparametrized_sample(self,
                              num_samples=1,
                              output_log_likelihoods=False,
                              detach_grad=False
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(self.model.num_times(), num_samples, self.model.num_strains())
        )

        # ======= Reparametrization
        samples = torch.stack([
            self.reparam_networks[t].forward(std_gaussian_samples[t, :, :])
            for t in range(self.model.num_times())
        ], dim=0)  # (T x N x S)

        if detach_grad:
            samples = samples.detach()

        if output_log_likelihoods:
            return samples, self.reparametrized_sample_log_likelihoods(samples)
        else:
            return samples

    def reparametrized_sample_log_likelihoods(self, samples: torch.Tensor):
        # input is (T x N x S)
        n_samples = samples.size()[1]
        ans = torch.zeros(size=(n_samples,), requires_grad=True, device=cfg.torch_cfg.device)
        for t in range(self.model.num_times()):
            samples_t = samples[t]
            linear = self.reparam_networks[t]
            try:
                w = linear.cholesky_part
                log_likelihood_t = torch.distributions.MultivariateNormal(
                    loc=linear.bias,
                    scale_tril=w
                ).log_prob(samples_t)
            except ValueError:
                logger.error(f"Problem while computing log MV log-likelihood of time index {t}.")
                raise
            ans = ans + log_likelihood_t
        return ans

    def log_likelihood(self, samples: torch.Tensor) -> float:
        if len(samples.size()) == 2:
            r, c = samples.size()
            samples = samples.view(r, 1, c)
        return super().log_likelihood(samples)


class GaussianPosteriorTimeCorrelation(AbstractReparametrizedPosterior):
    def __init__(self, model: GenerativeModel):
        """
        Mean-field assumption:
        1) Parametrize X_1, X_2, ..., X_S as independent T-dimensional gaussians (one per strain).
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        :param model: The generative model to use.
        """
        # Check: might need this to be a matrix, not a vector.
        self.model = model

        # ========== Reparametrization network (standard Gaussians -> nonstandard Gaussians)
        self.reparam_networks: Dict[int, torch.nn.Module] = dict()

        for s_idx in range(self.model.num_strains()):
            linear_layer = TrilLinear(
                n_features=self.model.num_times(),
                bias=True,
                device=cfg.torch_cfg.device
            )
            init_diag(linear_layer.weight, scale=1.0)
            torch.nn.init.constant_(linear_layer.bias, 0)
            self.reparam_networks[s_idx] = linear_layer
        self.parameters = []
        for network in self.reparam_networks.values():
            self.parameters += network.parameters()

        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def trainable_parameters(self) -> List[Parameter]:
        return self.parameters

    def mean(self) -> torch.Tensor:
        return torch.stack([
            self.reparam_networks[s].bias.detach()
            for s in range(self.model.num_strains())
        ], dim=1)

    def reparametrized_sample(self,
                              num_samples=1,
                              output_log_likelihoods=False,
                              detach_grad=False
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(self.model.num_strains(), num_samples, self.model.num_times()),
        )

        # ======= Reparametrization
        samples = torch.stack([
            self.reparam_networks[s_idx].forward(std_gaussian_samples[s_idx, :, :])
            for s_idx in range(self.model.num_strains())
        ], dim=0)

        if detach_grad:
            samples = samples.detach()

        if output_log_likelihoods:
            return samples.transpose(0, 2), self.reparametrized_sample_log_likelihoods(samples)
        else:
            return samples.transpose(0, 2)

    def reparametrized_sample_log_likelihoods(self, samples: torch.Tensor):
        n_samples = samples.size()[1]
        ans = torch.zeros(size=(n_samples,), requires_grad=True, device=cfg.torch_cfg.device)
        for s in range(self.model.num_strains()):
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

    def log_likelihood(self, samples: torch.Tensor) -> float:
        if len(samples.size()) == 2:
            r, c = samples.size()
            samples = samples.view(r, 1, c)
        return super().log_likelihood(samples)
