from typing import List, Optional, Callable, Type, Dict, Any

import numba
import numpy as np
import scipy.special
import torch

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads

from chronostrain.config import create_logger

from .. import AbstractModelSolver
from .posteriors import *
from .util import log_softmax, log_matmul_exp, psis_smooth_ratios
from .solver import BBVISolver
from ..vi import GaussianPosteriorFullCorrelation

logger = create_logger(__name__)


class BBVISolverFullPosterior(AbstractModelSolver):

    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 read_batch_size: int = 5000,
                 num_cores: int = 1,
                 partial_correlation_type: str = 'strain'):
        logger.info("Initializing full correlation solver (BBVI + Importance weighted estimation)")
        AbstractModelSolver.__init__(
            self, model, data, db, num_cores=num_cores
        )
        self.partial_solver = BBVISolver(
            model, data, db, read_batch_size, num_cores,
            correlation_type=partial_correlation_type
        )
        self.posterior: GaussianPosteriorFullCorrelation = None

    def prior_ll(self, x_samples: torch.Tensor) -> torch.Tensor:
        return self.model.log_likelihood_x(X=x_samples).detach()

    def data_ll(self, x_samples: torch.Tensor) -> torch.Tensor:
        ans = torch.zeros(size=(x_samples.shape[1],), device=x_samples.device)
        for t_idx in range(self.model.num_times()):
            log_y_t = log_softmax(x_samples, t=t_idx)
            for batch_lls in self.partial_solver.batches[t_idx]:
                batch_matrix = log_matmul_exp(log_y_t, batch_lls).detach()
                ans = ans + torch.sum(batch_matrix, dim=1)
        return ans

    def solve(self,
              optimizer_class: Type[torch.optim.Optimizer],
              optimizer_args: Dict[str, Any],
              num_epochs: int = 1,
              iters: int = 4000,
              num_bbvi_samples: int = 500,
              num_importance_samples: int = 5000,
              min_lr: float = 1e-4,
              lr_decay_factor: float = 0.25,
              lr_patience: int = 10,
              callbacks: Optional[List[Callable[[int, torch.Tensor, float], None]]] = None):
        """
        :param optimizer_class:
        :param optimizer_args:
        :param num_epochs:
        :param iters:
        :param num_bbvi_samples:
        :param num_importance_samples: The number of samples to use for importance sampling to estimate full posterior.
        :param min_lr:
        :param lr_decay_factor:
        :param lr_patience:
        :param callbacks:
        :return:
        """

        """First, solve using the mean-field factorized posteriors."""
        self.partial_solver.solve(
            optimizer_class, optimizer_args,
            num_epochs, iters, num_bbvi_samples,
            min_lr, lr_decay_factor, lr_patience,
            callbacks
        )

        """Next, estimate the true posterior covariance."""
        logger.debug("Performing importance-weighted estimation of parameters from learned posterior.")

        # (T x N x S)
        x_samples = self.partial_solver.posterior.reparametrized_sample(num_samples=num_importance_samples).detach()

        # length N
        approx_posterior_ll = self.partial_solver.posterior.reparametrized_sample_log_likelihoods(x_samples).detach()
        forward_ll = self.prior_ll(x_samples).detach() + self.data_ll(x_samples).detach()

        # normalize (we don't know normalization constant of true posterior).
        log_importance_ratios = forward_ll - approx_posterior_ll
        log_importance_ratios = log_importance_ratios.cpu().numpy()
        log_importance_ratios = log_importance_ratios - scipy.special.logsumexp(log_importance_ratios)
        log_smoothed_ratios, k_hat = psis_smooth_ratios(log_importance_ratios)
        logger.debug(f"Estimated Pareto k-hat: {k_hat}")
        if k_hat > 0.7:
            # Extremely large number of MCMC samples are needed for stable gradient estimates!
            logger.warning("Pareto k-hat estimate exceeds safe threshold (0.7). "
                           "Gradient estimates may have been unreliable in this regime.")

        # Importance sampling of mean and covariance matrix.
        total_sz = self.model.num_times() * self.model.num_strains()
        x_samples = x_samples.transpose(0, 1).reshape(num_importance_samples, total_sz)  # N x (TS)

        posterior_mean = torch.sum(
            x_samples * torch.exp(
                torch.unsqueeze(torch.tensor(log_smoothed_ratios, device=x_samples.device), dim=1)
            ),
            dim=0
        )
        posterior_var_scaling = torch.tensor(
            x_samples * torch.unsqueeze(
                torch.exp(0.5 * torch.tensor(log_smoothed_ratios, device=x_samples.device)),
                dim=1  # (N x 1)
            ),
            device=x_samples.device
        )
        self.posterior = GaussianPosteriorFullCorrelation(
            self.model.num_strains(), self.model.num_times(), posterior_mean, posterior_var_scaling
        )
        logger.debug("Finished computing importance-weighted mean/covariance.")
