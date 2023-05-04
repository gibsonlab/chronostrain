from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.math.matrices import log_mm_exp

from .base import AbstractADVISolver
from .posteriors import GaussianWithGlobalZerosPosterior, GaussianWithLocalZeros, GaussianWithGlobalZerosPosteriorSparsified
from ...subroutines.likelihoods import DataLikelihoods

from chronostrain.logging import create_logger
from chronostrain.model.zeros.gumbel import PopulationGlobalZeros, \
    PopulationLocalZeros  # TODO replace this with upper-level package import.

logger = create_logger(__name__)


class ADVIGaussianZerosSolver(AbstractADVISolver):
    """
    A basic implementation of ADVI estimating the posterior p(X|R), with fragments
    F marginalized out.
    """

    def __init__(self,
                 model: GenerativeModel,
                 zero_model: PopulationGlobalZeros,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 read_batch_size: int = 5000,
                 correlation_type: str = "full"):
        logger.info("Initializing solver with Gaussian posterior")
        if correlation_type == "full":
            # posterior = GaussianWithGlobalZerosPosterior(model, zero_model, 1e-5)
            posterior = GaussianWithGlobalZerosPosteriorSparsified(model, zero_model, 1e-5)
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(correlation_type))

        super().__init__(
            model=model,
            data=data,
            db=db,
            posterior=posterior,
            read_batch_size=read_batch_size
        )
        self.zero_model = zero_model

        self.log_total_marker_lens = torch.tensor([
            [np.log(sum(len(m) for m in strain.markers))]
            for strain in self.model.bacteria_pop.strains
        ], device=self.device)  # (S x 1)

    def advance_epoch(self):
        super().advance_epoch()
        self.posterior.inv_temp += 0.1

    def elbo(self,
             samples: Tuple[Tensor, Tensor]
             ) -> Iterator[Tensor]:
        """
        Computes the ADVI approximation to the ELBO objective, holding the read-to-fragment posteriors
        fixed. The entropic term is computed in closed-form, while the cross-entropy is given a monte-carlo estimate.

        The formula is:
            ELBO = E_Q(log P - log Q)
                = E_{X~Q}(log P(X) + P(R|X)) - E_{X~Qx}(log Q(X))
                = E_{X~Q}(log P(X) + P(R|X)) + H(Q)

        :param samples: A tuple of tensors.
            [0]: (T x N x S) tensor, representing the gaussian samples.
            [1]: (T x N x S) tensor, representing the Gumbel samples.
            [2]: (T-1 x N x S) tensor, representing the correlation Gumbels.
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
        import time

        gaussian_samples, smooth_log_zeros = samples
        start = time.time()
        yield self.model.log_likelihood_x(X=gaussian_samples).mean()  # Model Gaussian LL
        logger.debug("1: {} seconds".format(time.time() - start))

        start = time.time()
        zero_ll = self.zero_model.log_likelihood(smooth_log_zeros).mean()
        if zero_ll.requires_grad:
            yield zero_ll
        logger.debug("2: {} seconds".format(time.time() - start))

        start = time.time()
        yield self.posterior.entropy()
        logger.debug("3: {} seconds".format(time.time() - start))

        for t_idx in range(self.model.num_times()):
            start = time.time()
            log_y_t = self.model.log_latent_conversion(gaussian_samples[t_idx] + smooth_log_zeros)

            # Iterate over reads at timepoint t, pre-divided into batches.
            for batch_idx, batch_lls in enumerate(self.batches[t_idx]):
                batch_sz = batch_lls.shape[1]

                # Average of (N x R_batch) entries, we only want to divide by 1/N and not 1/(N*R_batch)
                # [but still call torch.mean() for numerical stability instead of torch.sum()]
                data_ll = batch_sz * torch.mean(
                    # (N x S) @ (S x R_batch)   ->  Raw data likelihood (up to proportionality)
                    log_mm_exp(log_y_t, batch_lls)
                )
                yield data_ll
            logger.debug("4 t={}: {} seconds".format(t_idx, time.time() - start))

            # (N x S) @ (S x 1)
            # -> Approx. conditional correction term for conditioning on markers. (note the minus sign)
            start = time.time()
            correction = len(self.data.time_slices[t_idx]) * torch.mean(
                -log_mm_exp(log_y_t, self.log_total_marker_lens)
            )
            yield correction
            logger.debug("5 t={}: {} seconds".format(t_idx, time.time() - start))

        # log_zeros = self.zero_model.smooth_log_zeroes_of_gumbels(gumbels, gumbels_between, self.zeros_inv_temperature)
        # for t_idx in range(self.model.num_times()):
        #     log_y_t = self.model.log_latent_conversion(gaussian_samples[t_idx] + log_zeros[t_idx])  # (N x S)
        #
        #     # Iterate over reads at timepoint t, pre-divided into batches.
        #     for batch_idx, batch_lls in enumerate(self.batches[t_idx]):
        #         batch_sz = batch_lls.shape[1]
        #
        #         # Average of (N x R_batch) entries, we only want to divide by 1/N and not 1/(N*R_batch)
        #         # [but still call torch.mean() for numerical stability instead of torch.sum()]
        #         data_ll = batch_sz * torch.mean(
        #             # (N x S) @ (S x R_batch)   ->  Raw data likelihood (up to proportionality)
        #             log_mm_exp(log_y_t, batch_lls)
        #         )
        #         yield data_ll
        #
        #     # (N x S) @ (S x 1)
        #     # -> Approx. conditional correction term for conditioning on markers. (note the minus sign)
        #     correction = len(self.data.time_slices[t_idx]) * torch.mean(
        #         -log_mm_exp(log_y_t, self.log_total_marker_lens)
        #     )
        #     yield correction

    def data_ll(self, x_samples: Tensor) -> Tensor:
        raise NotImplementedError()

    def model_ll(self, x_samples: Tensor):
        raise NotImplementedError()


class ADVIGaussianLocalZerosSolver(AbstractADVISolver):
    """
    A basic implementation of ADVI estimating the posterior p(X|R), with fragments
    F marginalized out.
    """

    def __init__(self,
                 model: GenerativeModel,
                 zero_model: PopulationLocalZeros,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 read_batch_size: int = 5000,
                 correlation_type: str = "full"):
        logger.info("Initializing solver with Gaussian posterior (Time-localized zeros)")
        if correlation_type == "full":
            posterior = GaussianWithLocalZeros(model, zero_model)
            self.inv_temp = 1e-5
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(correlation_type))

        super().__init__(
            model=model,
            data=data,
            db=db,
            posterior=posterior,
            read_batch_size=read_batch_size
        )
        self.zero_model = zero_model

        self.log_total_marker_lens = torch.tensor([
            [np.log(sum(len(m) for m in strain.markers))]
            for strain in self.model.bacteria_pop.strains
        ], device=self.device)  # (S x 1)

    def advance_epoch(self):
        super().advance_epoch()
        self.inv_temp += 0.1

    def elbo(self,
             samples: Tuple[Tensor, Tensor, Tensor]
             ) -> Iterator[Tensor]:
        """
        Computes the ADVI approximation to the ELBO objective, holding the read-to-fragment posteriors
        fixed. The entropic term is computed in closed-form, while the cross-entropy is given a monte-carlo estimate.

        The formula is:
            ELBO = E_Q(log P - log Q)
                = E_{X~Q}(log P(X) + P(R|X)) - E_{X~Qx}(log Q(X))
                = E_{X~Q}(log P(X) + P(R|X)) + H(Q)

        :param samples: A tuple of tensors.
            [0]: (T x N x S) tensor, representing the gaussian samples.
            [1]: (T x N x S) tensor, representing the Gumbel samples.
            [2]: (T-1 x N x S) tensor, representing the correlation Gumbels.
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
        gaussian_samples, gumbels, gumbels_between = samples
        yield self.model.log_likelihood_x(X=gaussian_samples).mean()  # Model Gaussian LL
        yield self.zero_model.log_likelihood(gumbels, gumbels_between).mean()
        yield self.posterior.entropy()

        log_zeros = self.zero_model.smooth_log_zeroes_of_gumbels(gumbels, gumbels_between, self.inv_temp)
        for t_idx in range(self.model.num_times()):
            log_y_t = self.model.log_latent_conversion(gaussian_samples[t_idx] + log_zeros[t_idx])  # (N x S)

            # Iterate over reads at timepoint t, pre-divided into batches.
            for batch_idx, batch_lls in enumerate(self.batches[t_idx]):
                batch_sz = batch_lls.shape[1]

                # Average of (N x R_batch) entries, we only want to divide by 1/N and not 1/(N*R_batch)
                # [but still call torch.mean() for numerical stability instead of torch.sum()]
                data_ll = batch_sz * torch.mean(
                    # (N x S) @ (S x R_batch)   ->  Raw data likelihood (up to proportionality)
                    log_mm_exp(log_y_t, batch_lls)
                )
                yield data_ll

            # (N x S) @ (S x 1)
            # -> Approx. conditional correction term for conditioning on markers. (note the minus sign)
            correction = len(self.data.time_slices[t_idx]) * torch.mean(
                -log_mm_exp(log_y_t, self.log_total_marker_lens)
            )
            yield correction

    def data_ll(self, x_samples: Tensor) -> Tensor:
        raise NotImplementedError()

    def model_ll(self, x_samples: Tensor):
        raise NotImplementedError()
