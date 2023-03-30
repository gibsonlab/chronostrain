from typing import Iterator, Optional

import numpy as np
import torch

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.math.matrices import log_mm_exp

from .base import AbstractADVISolver
from .posteriors import *
from ...subroutines.likelihoods import DataLikelihoods

from chronostrain.logging import create_logger
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
            posterior = GaussianPosteriorTimeCorrelation(model)
        elif correlation_type == "strain":
            posterior = GaussianPosteriorStrainCorrelation(model)
        elif correlation_type == "full":
            posterior = GaussianPosteriorFullReparametrizedCorrelation(model)
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

        self.log_total_marker_lens = torch.tensor([
            [np.log(sum(len(m) for m in strain.markers))]
            for strain in self.model.bacteria_pop.strains
        ], device=self.device)  # (S x 1)

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
        
        To obtain monte carlo estimates of ELBO, need to be able to compute:
        1. p(x)
        2. q(x), or more directly the entropy H(Q) = -E_Q[Q(X)]
        3. p(f|x) --> implemented via sparse log-likelihood matmul

        To save memory on larger inputs, split the ELBO up into several pieces.
        """
        model_ll = self.model.log_likelihood_x(X=x_samples).mean()
        yield model_ll

        entropic = self.posterior.entropy()
        yield entropic

        for t_idx in range(self.model.num_times()):
            log_y_t = self.model.log_latent_conversion(x_samples[t_idx])  # (N x S)
            for batch_idx, batch_lls in enumerate(self.batches[t_idx]):
                batch_sz = batch_lls.shape[1]

                # Average of (N x R_batch) entries, we only want to divide by 1/N and not 1/(N*R_batch)
                data_ll = batch_sz * torch.mean(
                    # (N x S) @ (S x R_batch)   ->  Raw data likelihood (up to proportionality)
                    log_mm_exp(log_y_t, batch_lls)
                    # (N x S) @ (S x 1)   -> Approx. correction term for conditioning on markers.
                    - log_mm_exp(log_y_t, self.log_total_marker_lens)
                )
                yield data_ll

    def data_ll(self, x_samples: torch.Tensor) -> torch.Tensor:
        ans = torch.zeros(size=(x_samples.shape[1],), device=x_samples.device)
        for t_idx in range(self.model.num_times()):
            log_y_t = self.model.log_latent_conversion(x_samples[t_idx]).detach()
            for batch_lls in self.batches[t_idx]:
                batch_matrix = log_mm_exp(log_y_t, batch_lls) - log_mm_exp(log_y_t, self.log_total_marker_lens)
                ans = ans + torch.sum(batch_matrix, dim=1)
        return ans

    def model_ll(self, x_samples: torch.Tensor):
        return self.model.log_likelihood_x(X=x_samples).detach()
