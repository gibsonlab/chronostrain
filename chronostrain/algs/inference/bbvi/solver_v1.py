from typing import List, Iterator, Optional, Callable

import torch

from chronostrain.database import StrainDatabase
from chronostrain.model import GenerativeModel
from chronostrain.model.io import TimeSeriesReads

from chronostrain.config import cfg, create_logger
from chronostrain.util.sparse import SparseMatrix
from chronostrain.util.sparse.sliceable import BBVIOptimizedSparseMatrix
from .. import AbstractModelSolver
from .base import AbstractBBVI
from .posteriors import *
from .util import log_softmax, LogMMExpDenseSPModel, LogMMExpDenseSPModel_Async

logger = create_logger(__name__)


class BBVISolverV1(AbstractModelSolver, AbstractBBVI):
    """
    A basic implementation of BBVI estimating the posterior p(X|R), with fragments
    F marginalized out.
    """
    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 frag_chunk_size: int = 100,
                 num_cores: int = 1,
                 correlation_type: str = "time"):
        logger.info("Initializing V1 solver (Marginalized posterior X)")
        AbstractModelSolver.__init__(
            self,
            model,
            data,
            db,
            frag_chunk_size=frag_chunk_size,
            num_cores=num_cores
        )

        self.correlation_type = correlation_type
        if correlation_type == "time":
            posterior = GaussianPosteriorTimeCorrelation(model=model)
        elif correlation_type == "strain":
            posterior = GaussianPosteriorStrainCorrelation(model=model)
        elif correlation_type == "full":
            posterior = GaussianPosteriorFullCorrelation(model=model)
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(correlation_type))

        AbstractBBVI.__init__(
            self,
            posterior,
            device=cfg.torch_cfg.device
        )

        self.frag_chunk_size = frag_chunk_size
        self.frag_freq_logmmexp: List[List[LogMMExpDenseSPModel_Async]] = [[] for _ in range(model.num_times())]
        self.data_ll_logmmexp: List[List[LogMMExpDenseSPModel_Async]] = [[] for _ in range(model.num_times())]

        logger.debug("Initializing BBVI data structures.")
        if not cfg.model_cfg.use_sparse:
            raise NotImplementedError("BBVI only supports sparse data structures.")

        frag_freqs: SparseMatrix = self.model.fragment_frequencies_sparse  # F x S
        for t_idx in range(model.num_times()):
            # Sparsity transformation (R^F -> R^{Support}), matrix size = (F' x F)
            projector = self.data_likelihoods.projectors[t_idx]

            n_chunks = 0
            for sparse_chunk in BBVIOptimizedSparseMatrix.optimize_from_sparse_matrix(
                    projector.sparse_mul(frag_freqs),
                    row_chunk_size=self.frag_chunk_size
            ).chunks:
                n_chunks += 1
                self.frag_freq_logmmexp[t_idx].append(LogMMExpDenseSPModel_Async(sparse_chunk.t()))

            logger.debug(f"Divided {projector.rows} x {frag_freqs.columns} sparse matrix "
                         f"into {n_chunks} chunks.")

            # Prepare data likelihood chunks.
            self.data_ll_logmmexp[t_idx] = [
                # Trace via JIT for a speedup (we will re-use these many times.)
                LogMMExpDenseSPModel_Async(chunk)
                for chunk in self.data_likelihoods.matrices[t_idx].chunks
            ]

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

        # ======== log P(R|X) = log Σ_F P(R|F)P(F|X)
        for t_idx in range(self.model.num_times()):
            # =========== NEW IMPLEMENTATION: chunks
            log_softmax_xt = log_softmax(x_samples, t=t_idx)
            for chunk_idx, (ff_chunk, data_chunk) in enumerate(
                    zip(self.frag_freq_logmmexp[t_idx], self.data_ll_logmmexp[t_idx])
            ):
                log_lls = data_chunk.forward(  # (N x F_CHUNK) @ (F_CHUNK x R) -> (N x R)
                    ff_chunk.forward(log_softmax_xt)  # (N x S) @ (S x F_CHUNK) -> (N x F_CHUNK)
                )
                e = torch.sum(log_lls[torch.isfinite(log_lls)]) * (1 / n_samples)
                yield e

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
