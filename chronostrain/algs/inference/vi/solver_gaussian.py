from typing import *

import jax
import jax.numpy as np
from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.optimization import LossOptimizer

from .base import AbstractADVISolver, _GENERIC_SAMPLE_TYPE, _GENERIC_PARAM_TYPE, _GENERIC_GRAD_TYPE
from .posteriors import *
from .util import log_mm_exp

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
        self.precompile_elbo()

    def precompile_elbo(self):
        from chronostrain.algs.inference.vi.posteriors.util import tril_linear_transform_with_bias
        n_times = self.model.num_times()
        n_data = np.expand_dims(
            np.array([len(self.data.time_slices[t_idx]) for t_idx in range(self.model.num_times())]),  # length T
            axis=1
        )
        log_total_marker_lens = np.array([
            np.log(sum(len(m) for m in strain.markers))
            for strain in self.model.bacteria_pop.strains
        ])  # length S: total marker nucleotide length of each strain

        # @jax.jit
        def _elbo(params, rand_samples):
            # print("***************************")
            # entropic
            elbo = params['diag_weights'].sum()
            # print(params['diag_weights'].sum())

            # model ll

            # x = tril_linear_transform_with_bias(
            #     params['tril_weights'],
            #     np.exp(params['diag_weights']),
            #     params['bias'],
            #     rand_samples['std_gaussians']
            # )
            # print(x)
            # x = x.reshape(n_times, rand_samples['std_gaussians'].shape[0], self.num_strains)

            x = self.posterior.reparametrize(rand_samples, params)
            # print(x)
            # print(self.model.log_likelihood_x(x=x).shape)
            # print(self.model.log_likelihood_x(x=x).mean())
            elbo += self.model.log_likelihood_x(x=x).mean()
            # print(self.model.log_likelihood_x(x=x).mean())

            # print("sum so far: {}".format(elbo))

            # data ll
            log_y = jax.nn.log_softmax(x, axis=-1)
            for t_idx in range(n_times):  # this loop is OK to flatten via JIT
                # (1 x N x S)
                log_y_t = jax.lax.dynamic_slice_in_dim(log_y, start_index=t_idx, slice_size=1, axis=0)

                for batch_idx, batch_lls in enumerate(self.batches[t_idx]):
                    batch_sz = batch_lls.shape[1]
                    elbo += batch_sz * log_mm_exp(log_y_t, batch_lls).mean()
                    # print("t = {}, batch = {} --> {}".format(
                    #         t_idx,
                    #         batch_idx,
                    #         batch_sz * log_mm_exp(log_y_t, batch_lls).mean()
                    # ))

            elbo += np.mean(-n_data * jax.scipy.special.logsumexp(log_y + log_total_marker_lens, axis=-1))
            return elbo

        self.elbo_grad = jax.value_and_grad(_elbo, argnums=0)






    #     self.precompile_elbo_components()
    #
    # # noinspection PyAttributeOutsideInit
    # def precompile_elbo_components(self):
    #     logger.info("Invoking JIT. Note that compilation doesn't actually occur until first invocation. "
    #                 "Thus, runtimes from the first iteration will not be accurate.")
    #     @jax.jit
    #     def entropy(params):
    #         diag_log_weights = params['diag_weights']
    #         return diag_log_weights.sum()
    #     self.entropy_grad = jax.value_and_grad(entropy, argnums=0)
    #
    #     @jax.jit
    #     def model_ll(params, rand_samples):
    #         x = self.posterior.reparametrize(rand_samples, params)
    #         return self.model.log_likelihood_x(x=x).mean()
    #     self.model_ll_grad = jax.value_and_grad(model_ll, argnums=0)
    #
    #     self.data_ll_grad: List[List[Callable]] = []
    #     self.conditional_correction_t_grad = []
    #     for t_idx in range(self.model.num_times()):
    #         data_ll_t_grad = []
    #         n_data = len(self.data.time_slices[t_idx])
    #         for batch_idx, batch_lls in enumerate(self.batches[t_idx]):
    #             batch_sz = batch_lls.shape[1]
    #
    #             @jax.jit
    #             def data_ll_t_fn(params, rand_samples):
    #                 x = self.posterior.reparametrize(rand_samples, params)
    #                 log_y_t = jax.nn.log_softmax(x, axis=-1)  # TODO: this is not right; only take index t
    #                 # (N x S) @ (S x R_batch)   ->  Raw data likelihood (up to proportionality)
    #                 return batch_sz * log_mm_exp(log_y_t, batch_lls).mean()
    #             data_ll_t_grad.append(
    #                 jax.value_and_grad(data_ll_t_fn)
    #             )
    #         self.data_ll_grad.append(data_ll_t_grad)
    #
    #         @jax.jit
    #         def conditional_correction_t_fn(params, rand_samples):
    #             x = self.posterior.reparametrize(rand_samples, params)
    #             log_y_t = jax.nn.log_softmax(x, axis=1)  # TODO: this is not right; only take index t
    #             # (N x S) @ (S x 1)   -> Approx. correction term for conditioning on markers.
    #             return -n_data * jax.scipy.special.logsumexp(log_y_t + self.log_total_marker_lens, axis=-1).mean()
    #         self.conditional_correction_t_grad.append(
    #             jax.value_and_grad(conditional_correction_t_fn)
    #         )

    def elbo_with_grad(
            self,
            params: _GENERIC_PARAM_TYPE,
            random_samples: _GENERIC_SAMPLE_TYPE
    ) -> Tuple[np.ndarray, _GENERIC_GRAD_TYPE]:
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
        return self.elbo_grad(params, random_samples)
        # yield self.entropy_grad(params)
        # yield self.model_ll_grad(params, random_samples)
        # for data_ll_t_grad in self.data_ll_grad:
        #     for t_batch_fn in data_ll_t_grad:
        #         yield t_batch_fn(params, random_samples)
        # for corrs_t_fn in self.conditional_correction_t_grad:
        #     yield corrs_t_fn(params, random_samples)

    # def elbo_with_grad(
    #         self,
    #         params: _GENERIC_PARAM_TYPE,
    #         random_samples: _GENERIC_SAMPLE_TYPE
    # ) -> Tuple[np.ndarray, _GENERIC_GRAD_TYPE]:
    #     total_elbo = None
    #     # noinspection PyTypeChecker
    #     total_grad: _GENERIC_GRAD_TYPE = None
    #     for elbo_chunk, elbo_chunk_grad in self.accumulate_elbos(params, random_samples):
    #         if total_elbo is None:
    #             total_elbo = elbo_chunk
    #             total_grad = elbo_chunk_grad
    #         else:
    #             total_elbo += elbo_chunk
    #             for k, g in total_grad.items():
    #                 total_grad[k] = total_grad[k] + elbo_chunk_grad[k]
    #     if np.isinf(total_elbo):
    #         raise ValueError("Infinite ELBO objective encountered.")
    #     return total_elbo, total_grad
