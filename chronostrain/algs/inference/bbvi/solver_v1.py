from typing import List, Iterator, Optional, Callable

import numpy as np
import torch

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads

from chronostrain.config import cfg, create_logger
from chronostrain.util.math import log_spspmm_exp
from chronostrain.util.sparse.sliceable import ColumnSectionedSparseMatrix, RowSectionedSparseMatrix
from .. import AbstractModelSolver
from .base import AbstractBBVI
from .posteriors import *
from .util import log_softmax, LogMMExpModel

logger = create_logger(__name__)


def divide_columns_into_batches(x: torch.Tensor, read_batch_size: int):
    columns = torch.randperm(x.shape[1], device=cfg.torch_cfg.device)
    for i in range(np.ceil(x.shape[1] / read_batch_size)):
        left_idx = i * read_batch_size
        right_idx = np.min(x.shape[1], (i + 1) * read_batch_size)
        yield x[:, columns[left_idx:right_idx]]


class BBVISolverV1(AbstractModelSolver, AbstractBBVI):
    """
    A basic implementation of BBVI estimating the posterior p(X|R), with fragments
    F marginalized out.
    """
    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 read_batch_size: int = 5000,
                 num_cores: int = 1,
                 correlation_type: str = "time"):
        logger.info("Initializing V1 solver (Marginalized posterior X)")
        AbstractModelSolver.__init__(
            self,
            model,
            data,
            db,
            num_cores=num_cores
        )

        self.correlation_type = correlation_type
        if correlation_type == "time":
            posterior = GaussianPosteriorTimeCorrelation(
                num_strains=model.num_strains(),
                num_times=model.num_times()
            )
        elif correlation_type == "strain":
            posterior = GaussianPosteriorStrainCorrelation(
                num_strains=model.num_strains(),
                num_times=model.num_times()
            )
        elif correlation_type == "full":
            posterior = GaussianPosteriorFullCorrelation(
                num_strains=model.num_strains(),
                num_times=model.num_times()
            )
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(correlation_type))

        AbstractBBVI.__init__(
            self,
            posterior,
            device=cfg.torch_cfg.device
        )

        logger.debug("Initializing BBVI data structures.")
        if not cfg.model_cfg.use_sparse:
            raise NotImplementedError("BBVI only supports sparse data structures.")

        # (S x R) matrices: Contains P(R = r | S = s) for each read r, strain s.
        self.strain_read_ll_model_batches: List[List[LogMMExpModel]] = [
            [] for _ in range(model.num_times())
        ]

        # Precompute this (only possible in V1).
        logger.debug("Precomputing likelihood products.")
        for t_idx in range(model.num_times()):
            data_ll_t = self.data_likelihoods.matrices[t_idx]  # F x R

            projector = self.data_likelihoods.projectors[t_idx]
            strain_read_lls_t = log_spspmm_exp(
                ColumnSectionedSparseMatrix.from_sparse_matrix(
                    projector.sparse_mul(self.model.fragment_frequencies_sparse).t()
                ),  # (S x F_), note the transpose!
                data_ll_t  # (F_ x R)
            )  # (S x R)

            for batch_matrix in divide_columns_into_batches(strain_read_lls_t, read_batch_size):
                self.strain_read_ll_model_batches[t_idx].append(
                    LogMMExpModel(batch_matrix)
                )

    def elbo(self,
             x_samples: torch.Tensor,
             posterior_sample_lls: torch.Tensor
             ) -> Iterator[torch.Tensor]:
        """
        Computes the monte-carlo approximation to the ELBO objective, holding the read-to-fragment posteriors fixed.

        The formula is:
            ELBO = E_Q(log P - log Q)
                = E_{X~Q}(log P(X) + P(R|X)) - E_{X~Qx}(log Q(X))

        :param x_samples: A (T x N x S) tensor, where T = # of timepoints, N = # of samples, S = # of strains.
        :param posterior_sample_lls: A length-N (one-dimensional) tensor of the joint log-likelihood
            each (T x S) slice.
        :return: An estimate of the ELBO, using the provided samples via the above formula.
        """

        """
        ELBO original formula:
            E_Q[P(X)] - E_Q[Q(X)] + E_{F ~ phi} [log P(F | X)]

        To save memory on larger frag spaces, split the ELBO up into several pieces.
        """
        n_samples = x_samples.size()[1]
        # ======== -log Q(X), monte-carlo
        yield posterior_sample_lls.sum() * (-1 / n_samples)

        # ======== log P(X)
        model_gaussian_log_likelihoods = self.model.log_likelihood_x(X=x_samples)
        yield model_gaussian_log_likelihoods.sum() * (1 / n_samples)

        # ======== log P(R|X) = log Σ_S P(R|S)P(S|X)
        for t_idx in range(self.model.num_times()):
            log_softmax_xt = log_softmax(x_samples, t=t_idx)
            for batch_model in self.strain_read_ll_model_batches[t_idx]:
                yield (1 / n_samples) * torch.sum(batch_model.forward(log_softmax_xt))

    def solve(self,
              optimizer: torch.optim.Optimizer,
              lr_scheduler,
              num_epochs: int = 1,
              iters: int = 4000,
              num_samples: int = 8000,
              min_lr: float = 1e-4,
              callbacks: Optional[List[Callable[[int, torch.Tensor, float], None]]] = None):
        self.optimize(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            iters=iters,
            num_epochs=num_epochs,
            num_samples=num_samples,
            min_lr=min_lr,
            callbacks=callbacks
        )
