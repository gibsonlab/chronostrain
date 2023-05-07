from typing import *

import jax
import jax.numpy as np
from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.math.matrices import log_mm_exp
from chronostrain.util.optimization import LossOptimizer

from .base import AbstractADVISolver, _GENERIC_SAMPLE_TYPE, _GENERIC_PARAM_TYPE, _GENERIC_GRAD_TYPE
from .posteriors import *

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
                 optimizer: LossOptimizer,
                 read_batch_size: int = 5000,
                 correlation_type: str = "time"):
        logger.info("Initializing solver with Gaussian posterior")
        if correlation_type == "time":
            raise NotImplementedError("TODO implement this posterior for Jax.")
            # posterior = GaussianPosteriorTimeCorrelation(model)
        elif correlation_type == "strain":
            raise NotImplementedError("TODO implement this posterior for Jax.")
            # posterior = GaussianPosteriorStrainCorrelation(model)
        elif correlation_type == "full":
            posterior = GaussianPosteriorFullReparametrizedCorrelation(model)
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(correlation_type))

        super().__init__(
            model=model,
            data=data,
            db=db,
            posterior=posterior,
            optimizer=optimizer,
            read_batch_size=read_batch_size
        )

        self.log_total_marker_lens = np.array([
            np.log(sum(len(m) for m in strain.markers))
            for strain in self.model.bacteria_pop.strains
        ])  # total marker nucleotide length of each strain

        self.precompile_elbo_components()

    # noinspection PyAttributeOutsideInit
    def precompile_elbo_components(self):
        self.entropy_grad = jax.value_and_grad(self.posterior.entropy, argnums=0)

        @jax.jit
        def model_ll(params, rand_samples):
            x = self.posterior.reparametrize(rand_samples, params)
            self.model.log_likelihood_x(x=x).mean()
        self.model_ll_grad = jax.value_and_grad(model_ll, argnums=0)

        self.data_ll_grad: List[List[Callable]] = []
        self.conditional_correction_t_grad = []
        for t_idx in range(self.model.num_times()):
            data_ll_t_grad = []
            for batch_idx, batch_lls in enumerate(self.batches[t_idx]):
                batch_sz = batch_lls.shape[1]
                n_data = len(self.data.time_slices[t_idx])

                @jax.jit
                def data_ll_t_fn(params, rand_samples):
                    x = self.posterior.reparametrize(rand_samples, params)
                    log_y_t = jax.nn.log_softmax(x, axis=-1)
                    return batch_sz * log_mm_exp(log_y_t, batch_lls).mean()
                data_ll_t_grad.append(
                    jax.value_and_grad(data_ll_t_fn)
                )
            self.data_ll_grad.append(data_ll_t_grad)

            @jax.jit
            def conditional_correction_t_fn(params, rand_samples):
                x = self.posterior.reparametrize(rand_samples, params)
                log_y_t = jax.nn.log_softmax(x, axis=1)
                return n_data * -log_mm_exp(log_y_t, self.log_total_marker_lens).mean()
            self.conditional_correction_t_grad.append(
                jax.value_and_grad(conditional_correction_t_fn)
            )

    def accumulate_elbos(
            self,
            params: _GENERIC_PARAM_TYPE,
            random_samples: _GENERIC_SAMPLE_TYPE
    ) -> Iterator[np.ndarray, _GENERIC_GRAD_TYPE]:
        """
        Computes the ADVI approximation to the ELBO objective, holding the read-to-fragment posteriors
        fixed. The entropic term is computed in closed-form, while the cross-entropy is given a monte-carlo estimate.

        The formula is:
            ELBO = E_Q(log P - log Q)
                = E_{X~Q}(log P(X) + P(R|X)) - E_{X~Qx}(log Q(X))
                = E_{X~Q}(log P(X) + P(R|X)) + H(Q)

        :param params: The value of parameters to use for reparametrization.
        :param random_samples: A dict of the random samples before reparametrization.
        :return: An estimate of the ELBO, yielded piece by piece.
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
        yield self.entropy_grad(params)
        yield self.model_ll_grad(params, random_samples)

        for data_ll_t_grad in self.data_ll_grad:
            for t_batch_fn in data_ll_t_grad:
                yield t_batch_fn(params, random_samples)

        for corrs_t_fn in self.conditional_correction_t_grad:
            yield corrs_t_fn(params, random_samples)

    def elbo_with_grad(
            self,
            params: _GENERIC_PARAM_TYPE,
            random_samples: _GENERIC_SAMPLE_TYPE
    ) -> Tuple[np.ndarray, _GENERIC_GRAD_TYPE]:
        total_elbo = None
        # noinspection PyTypeChecker
        total_grad: _GENERIC_GRAD_TYPE = None
        for elbo_chunk, elbo_chunk_grad in self.accumulate_elbos(params, random_samples):
            if total_elbo is None:
                total_elbo = elbo_chunk
                total_grad = elbo_chunk_grad
            else:
                total_elbo += elbo_chunk
                for k, g in total_grad.items():
                    total_grad[k] = total_grad[k] + elbo_chunk_grad[k]
        return total_elbo, total_grad
