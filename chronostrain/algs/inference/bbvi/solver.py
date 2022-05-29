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
from ...subroutines.likelihoods import DataLikelihoods

logger = create_logger(__name__)


def divide_columns_into_batches(x: torch.Tensor, batch_size: int):
    permutation = torch.randperm(x.shape[1], device=cfg.torch_cfg.device)
    for i in range(0, x.shape[1], batch_size):
        indices = permutation[i:i+batch_size]
        yield x[:, indices]


class BBVISolver(AbstractModelSolver, AbstractBBVI):
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
                 correlation_type: str = "time",
                 precomputed_data_likelihoods: Optional[DataLikelihoods] = None):
        logger.info("Initializing V1 solver (Marginalized posterior X)")
        AbstractModelSolver.__init__(
            self,
            model,
            data,
            db,
            num_cores=num_cores,
            precomputed_data_likelihoods=precomputed_data_likelihoods
        )

        self.read_batch_size = read_batch_size
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
            logger.warning("Full correlation posterior for this solver may result in biased/unstable estimates.")
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
        self.strain_read_lls: List[torch.Tensor] = []
        self.batches: List[List[torch.Tensor]] = [
            [] for _ in range(model.num_times())
        ]
        self.total_reads: int = 0

        # Precompute this (only possible in V1).
        logger.debug("Precomputing likelihood products.")
        for t_idx in range(model.num_times()):
            data_ll_t = self.data_likelihoods.matrices[t_idx]  # F_ x R
            self.total_reads += data_ll_t.shape[1]

            projector = self.data_likelihoods.projectors[t_idx]
            strain_read_lls_t = log_spspmm_exp(
                ColumnSectionedSparseMatrix.from_sparse_matrix(
                    projector.sparse_mul(self.model.fragment_frequencies_sparse).t()
                ),  # (S x F_), note the transpose!
                data_ll_t  # (F_ x R)
            )  # (S x R)

            # Locate and filter out reads with no good alignments.
            bad_indices = set(
                float(x.cpu())
                for x in torch.where(torch.sum(~torch.isinf(strain_read_lls_t), dim=0) == 0)[0]
            )
            good_indices = [i for i in range(data_ll_t.shape[1]) if i not in bad_indices]
            if len(bad_indices) > 0:
                logger.warning("(t = {}) Found {} reads without good alignments: {}".format(
                    t_idx,
                    len(bad_indices),
                    [self.data[t_idx][int(i)].id for i in bad_indices]
                ))
                strain_read_lls_t = strain_read_lls_t[:, good_indices]

            self.strain_read_lls.append(strain_read_lls_t)

    def elbo(self,
             x_samples: torch.Tensor
             ) -> Iterator[torch.Tensor]:
        """
        Computes the monte-carlo approximation to the ELBO objective, holding the read-to-fragment posteriors fixed.

        The formula is:
            ELBO = E_Q(log P - log Q)
                = E_{X~Q}(log P(X) + P(R|X)) - E_{X~Qx}(log Q(X))

        :param x_samples: A (T x N x S) tensor, where T = # of timepoints, N = # of samples, S = # of strains.
        :return: An estimate of the ELBO, using the provided samples via the above formula.
        """

        """
        ELBO original formula:
            E_Q[P(X)] - E_Q[Q(X)] + E_{F ~ phi} [log P(F | X)]

        To save memory on larger frag spaces, split the ELBO up into several pieces.
        """
        n_samples = x_samples.size()[1]

        # ======== H(Q) = E_Q[-log Q(X)]
        entropic = self.posterior.entropy()

        # ======== E[log P(X)]
        model_gaussian_log_likelihoods = self.model.log_likelihood_x(X=x_samples)
        model_ll = torch.mean(model_gaussian_log_likelihoods)
        yield entropic + model_ll

        # ======== E[log P(R|X)] = E[log Σ_S P(R|S)P(S|X)]
        time_opt_ordering = np.random.permutation(list(range(self.model.num_times())))
        for t_idx in time_opt_ordering:
            log_y_t = log_softmax(x_samples, t=t_idx)
            for batch_lls in self.batches[t_idx]:
                # batch_ratio = batch_lls.shape[1] / self.total_reads
                yield torch.sum(
                    (1 / n_samples) * log_matmul_exp(log_y_t, batch_lls)
                )

    def advance_epoch(self):
        for t_idx in range(self.model.num_times()):
            self.batches[t_idx] = list(divide_columns_into_batches(self.strain_read_lls[t_idx], self.read_batch_size))

    def solve(self,
              optimizer_class: Type[torch.optim.Optimizer],
              optimizer_args: Dict[str, Any],
              num_epochs: int = 1,
              iters: int = 4000,
              num_samples: int = 8000,
              min_lr: float = 1e-4,
              lr_decay_factor: float = 0.25,
              lr_patience: int = 10,
              callbacks: Optional[List[Callable[[int, torch.Tensor, float], None]]] = None):
        """Idea: To encourage exploration, optimize in two rounds: Mean and Mean+Variance."""
        # logger.debug("Using three-round training strategy.")

        def do_optimize(opt, sched):
            self.optimize(
                optimizer=opt,
                lr_scheduler=sched,
                iters=iters,
                num_epochs=num_epochs,
                num_samples=num_samples,
                min_lr=min_lr,
                callbacks=callbacks
            )

        # # Round 1: mean only
        # logger.debug("Training round #1 of 3.")
        # optimizer_args['params'] = self.posterior.trainable_mean_parameters()
        # optimizer = optimizer_class(**optimizer_args)
        # lr_scheduler = ReduceLROnPlateauLast(
        #     optimizer,
        #     factor=lr_decay_factor,
        #     patience_horizon=lr_patience,
        #     patience_ratio=0.5,
        #     threshold=1e-4,
        #     threshold_mode='rel',
        #     mode='min'  # track (-ELBO) and decrease LR when it stops decreasing.
        # )
        # do_optimize(optimizer, lr_scheduler)
        #
        # # # Round 2: variance only
        # logger.debug("Training round #2 of 3.")
        # optimizer_args['params'] = self.posterior.trainable_variance_parameters()
        # optimizer = optimizer_class(**optimizer_args)
        # lr_scheduler = ReduceLROnPlateauLast(
        #     optimizer,
        #     factor=lr_decay_factor,
        #     patience_horizon=lr_patience,
        #     patience_ratio=0.5,
        #     threshold=1e-2,
        #     threshold_mode='rel',
        #     mode='min'  # track (-ELBO) and decrease LR when it stops decreasing.
        # )
        # do_optimize(optimizer, lr_scheduler)

        # Round 3: all parameters. TODO test just "round 3" next.
        logger.debug("Training round #3 of 3.")
        optimizer_args['params'] = self.posterior.trainable_parameters()
        optimizer = optimizer_class(**optimizer_args)
        lr_scheduler = ReduceLROnPlateauLast(
            optimizer,
            factor=lr_decay_factor,
            patience_horizon=lr_patience,
            patience_ratio=0.5,
            threshold=1e-2,
            threshold_mode='rel',
            mode='min'  # track (-ELBO) and decrease LR when it stops decreasing.
        )
        do_optimize(optimizer, lr_scheduler)
