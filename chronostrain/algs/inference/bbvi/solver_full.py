from typing import List, Iterator, Optional, Callable, Type, Dict, Any

import numpy as np
import torch

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads

from chronostrain.config import cfg, create_logger
from chronostrain.util.math import log_spspmm_exp
from chronostrain.util.optimization import ReduceLROnPlateauLast
from chronostrain.util.sparse.sliceable import ColumnSectionedSparseMatrix

from .. import AbstractModelSolver
from .base import AbstractBBVI
from .posteriors import *
from .util import log_softmax, log_matmul_exp
from .solver import BBVISolver

logger = create_logger(__name__)


class BBVISolverFullPosterior(AbstractModelSolver):

    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 read_batch_size: int = 5000,
                 num_cores: int = 1,
                 partial_correlation_type: str = 'strain'):
        AbstractModelSolver.__init__(
            self, model, data, db, num_cores=num_cores
        )

        self.partial_solver = BBVISolver(
            model, data, db, read_batch_size, num_cores,
            correlation_type=partial_correlation_type
        )

    def prior_ll(self, x_samples: torch.Tensor) -> torch.Tensor:
        return self.model.log_likelihood_x(X=x_samples).detach()

    def data_ll(self, x_samples: torch.Tensor) -> torch.Tensor:
        ans = torch.zeros(size=(x_samples.shape[1]), device=x_samples.device)
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
              num_estimation_samples: int = 5000,
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
        :param num_estimation_samples: The number of samples to use for importance sampling to estimate full posterior.
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
        # (T x N x S)
        x_samples = self.partial_solver.posterior.reparametrized_sample(num_samples=num_estimation_samples).detach()

        # length N
        approx_posterior_ll = self.partial_solver.posterior.reparametrized_sample_log_likelihoods(x_samples).detach()
        forward_ll = self.prior_ll(x_samples) + self.data_ll(x_samples)

        log_importance_ratios = forward_ll / approx_posterior_ll
        # normalize (we don't know normalization constant of true posterior).
        log_importance_ratios = log_importance_ratios / torch.sum(log_importance_ratios)


