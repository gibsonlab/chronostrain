from typing import Iterator, Optional

import numpy as np
import torch

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads

from chronostrain.config import create_logger
from .base import AbstractADVISolver
from .posteriors import *
from .util import log_softmax_t, log_matmul_exp
from ...subroutines.likelihoods import DataLikelihoods

logger = create_logger(__name__)


class ADVIGaussianSolver(AbstractADVISolver):
    """
    A basic implementation of ADVI estimating the posterior p(X|R), with fragments
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
        logger.info("Initializing solver with Gaussian posterior")
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
            posterior = GaussianPosteriorFullReparametrizedCorrelation(
                num_strains=model.num_strains(),
                num_times=model.num_times()
            )
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(correlation_type))

        super().__init__(
            model=model,
            data=data,
            db=db,
            posterior=posterior,
            read_batch_size=read_batch_size,
            num_cores=num_cores,
            precomputed_data_likelihoods=precomputed_data_likelihoods
        )

    def elbo(self,
             x_samples: torch.Tensor
             ) -> Iterator[torch.Tensor]:
        """
        Computes the ADVI approximation to the ELBO objective, holding the read-to-fragment posteriors
        fixed. The entropic term is computed in closed-form, while the cross-entropy is given a monte-carlo estimate.

        The formula is:
            ELBO = E_Q(log P - log Q)
                = E_{X~Q}(log P(X) + P(R|X)) - E_{X~Qx}(log Q(X))
                = E_{X~Q}(log P(X) + P(R|X)) + H(Q)

        :param x_samples: A (T x N x S) tensor, where T = # of timepoints, N = # of samples, S = # of strains.
        :return: An estimate of the ELBO, using the provided samples via the above formula.
        """

        """
        ELBO original formula:
            E_Q[P(X)] - E_Q[Q(X)] + E_{F ~ phi} [log P(F | X)]

        To save memory on larger frag spaces, split the ELBO up into several pieces.
        """

        # ======== E[log P(R|X)] = E[log Σ_S P(R|S)P(S|X)]
        for t_idx in np.random.permutation(self.model.num_times()):
            log_y_t = log_softmax_t(x_samples, t=t_idx)
            for batch_lls in self.batches[t_idx]:
                wt = batch_lls.shape[1] / self.total_reads

                # Average of (N x R_batch) entries, we only want to divide by 1/N and not 1/(N*R_batch)
                data_ll = batch_lls.shape[1] * torch.mean(log_matmul_exp(log_y_t, batch_lls))
                yield data_ll + wt * (self.posterior.entropy() + self.model.log_likelihood_x(X=x_samples).mean())

    def data_ll(self, x_samples: torch.Tensor) -> torch.Tensor:
        ans = torch.zeros(size=(x_samples.shape[1],), device=x_samples.device)
        for t_idx in range(self.model.num_times()):
            log_y_t = log_softmax_t(x_samples, t=t_idx)
            for batch_lls in self.batches[t_idx]:
                batch_matrix = log_matmul_exp(log_y_t, batch_lls).detach()
                ans = ans + torch.sum(batch_matrix, dim=1)
        return ans

    def model_ll(self, x_samples: torch.Tensor):
        return self.model.log_likelihood_x(X=x_samples).detach()

    def solve(self,
              optimizer_class,
              optimizer_args,
              num_epochs: int = 1,
              iters: int = 4000,
              num_samples: int = 8000,
              min_lr: float = 1e-4,
              lr_decay_factor: float = 0.25,
              lr_patience: int = 10,
              callbacks = None):
        from chronostrain.util.optimization import ReduceLROnPlateauLast
        from chronostrain.algs.inference.bbvi.posteriors.gaussians import GaussianPosteriorFullReparametrizedCorrelation

        def _optimize(params):
            optimizer_args['params'] = params
            optimizer = optimizer_class(**optimizer_args)
            lr_scheduler = ReduceLROnPlateauLast(
                optimizer,
                factor=lr_decay_factor,
                patience_horizon=lr_patience,
                patience_ratio=0.5,
                threshold=1e-4,
                threshold_mode='rel',
                mode='min'  # track (-ELBO) and decrease LR when it stops decreasing.
            )
            self.optimize(
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                iters=iters,
                num_epochs=num_epochs,
                num_samples=num_samples,
                min_lr=min_lr,
                callbacks=callbacks
            )
            self.diagnostic()

        assert isinstance(self.posterior, GaussianPosteriorFullReparametrizedCorrelation)
        _optimize(self.posterior.trainable_mean_parameters())
        _A = self.posterior.left_linear_transform()
        print(_A)
        print(_A @ _A.transpose(0, 1))

        _optimize(self.posterior.trainable_variance_parameters())
        _A = self.posterior.left_linear_transform()
        print(_A)
        print(_A @ _A.transpose(0, 1))

        _optimize(self.posterior.trainable_parameters())
        _A = self.posterior.left_linear_transform()
        print(_A)
        print(_A @ _A.transpose(0, 1))
