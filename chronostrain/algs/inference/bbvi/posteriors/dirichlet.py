from pathlib import Path
from typing import Union, Tuple, List

import torch
from torch.distributions import Dirichlet, Normal
from torch.nn import Parameter

from .base import AbstractReparametrizedPosterior

from chronostrain.config import cfg, create_logger
logger = create_logger(__name__)


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

        self.concentrations = torch.nn.Parameter(
            torch.ones(
                self.num_times, self.num_strains,
                device=cfg.torch_cfg.device
            ),
            requires_grad=True
        )

        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def trainable_parameters(self) -> List[Parameter]:
        return [self.concentrations]

    def mean(self) -> torch.Tensor:
        return torch.detach(self.concentrations / torch.sum(self.concentrations, dim=1, keepdim=True))

    def entropy(self) -> torch.Tensor:
        return Dirichlet(self.concentrations).entropy().sum()

    def gaussian_approximation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Laplace gaussian-softmax reparametrized approximation, in the style of
        Srivasatava, Sutton: AUTOENCODING VARIATIONAL INFERENCE FOR TOPIC MODELS
        https://arxiv.org/pdf/1703.01488.pdf

        :return: The (T x S) mean and (T x S) standard deviation parameters.
        """
        mean = torch.log(self.concentrations) - (1 / self.num_strains) * torch.sum(self.concentrations, dim=1, keepdim=True)

        inv_conc = torch.reciprocal(self.concentrations)

        # Approximation uses diagonal covariance.
        scaling = torch.sqrt(
            (1 - (2 / self.num_strains)) * inv_conc
            + (1 / self.num_strains ** 2) * torch.sum(inv_conc, dim=1, keepdim=True)
        )
        return mean, scaling

    def reparametrized_sample(self,
                              num_samples=1
                              ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        std_gaussian_samples = self.standard_normal.sample(
            sample_shape=(self.num_times, num_samples, self.num_strains)
        )
        mean, scaling = self.gaussian_approximation()
        s = torch.softmax(
            torch.unsqueeze(mean, 1) + torch.unsqueeze(scaling, 1) * std_gaussian_samples,
            dim=2
        )
        return s

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        return self.reparametrized_sample(num_samples=num_samples).detach()

    def log_likelihood(self, dirichlet_samples: torch.Tensor):
        # WARNING: Not to be used for autograd in ADVI!
        return Dirichlet(self.concentrations).log_prob(dirichlet_samples.transpose(0, 1)).sum(dim=1)

    def save(self, path: Path):
        torch.save(self.concentrations.detach().cpu(), path)

    @staticmethod
    def load(path: Path, num_strains: int, num_times: int) -> 'ReparametrizedDirichletSolver':
        posterior = ReparametrizedDirichletPosterior(num_strains, num_times)
        concs = torch.load(path)
        assert isinstance(concs, torch.Tensor)
        posterior.concentrations = concs
        return posterior
