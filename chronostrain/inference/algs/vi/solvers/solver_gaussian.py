from typing import Protocol, List, Tuple

import jax
import jax.numpy as np
import numpy as cnp
from chronostrain.database import StrainDatabase
from chronostrain.model import *
from chronostrain.util.optimization import LossOptimizer

from chronostrain.inference.algs.vi.base.advi import AbstractADVISolver
from chronostrain.inference.algs.vi.base.constants import GENERIC_SAMPLE_TYPE, GENERIC_PARAM_TYPE, GENERIC_GRAD_TYPE
from chronostrain.inference.algs.vi.base.posterior import AbstractReparametrizedPosterior
from ..posteriors import GaussianPosteriorFullReparametrizedCorrelation
from chronostrain.inference.algs.vi.base.util import log_mm_exp

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class BiasInitializer(Protocol):
    def __call__(self, population: Population, times: List[float]) -> np.ndarray:
        pass


class ADVIGaussianSolver(AbstractADVISolver):
    """
    A basic implementation of ADVI estimating the posterior p(X|R), with fragments
    F marginalized out.

    Computes the ADVI approximation to the ELBO objective, holding the read-to-fragment posteriors
    fixed. The entropic term is computed in closed-form, while the cross-entropy is given a monte-carlo estimate.

    The formula is:
        ELBO = E_Q(log P - log Q)
            = E_{X~Q}(log P(X) + P(R|X)) - E_{X~Qx}(log Q(X))
            = E_{X~Q}(log P(X) + P(R|X)) + H(Q)
    """

    def __init__(self,
                 gaussian_prior: AbundanceGaussianPrior,
                 error_model: AbstractErrorModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 optimizer: LossOptimizer,
                 prune_strains: bool,
                 bias_initializer: BiasInitializer,
                 read_batch_size: int = 5000,
                 accumulate_gradients: bool = False,
                 correlation_type: str = "time",
                 dtype='bfloat16'):
        logger.info("Initializing solver with Gaussian posterior")
        self.dtype = dtype
        self.correlation_type = correlation_type
        self.bias_initializer = bias_initializer

        super().__init__(
            gaussian_prior=gaussian_prior,
            error_model=error_model,
            data=data,
            db=db,
            optimizer=optimizer,
            prune_strains=prune_strains,
            read_batch_size=read_batch_size
        )

        self.accumulate_gradients = accumulate_gradients
        if self.accumulate_gradients:
            self.precompile_elbo_pieces()
        else:
            self.precompile_elbo()

    def create_posterior(self) -> AbstractReparametrizedPosterior:
        if self.correlation_type == "time":
            raise NotImplementedError("TODO implement this posterior for Jax.")
            # posterior = GaussianPosteriorTimeCorrelation(model)
        elif self.correlation_type == "strain":
            raise NotImplementedError("TODO implement this posterior for Jax.")
            # posterior = GaussianPosteriorStrainCorrelation(model)
        elif self.correlation_type == "full":
            return GaussianPosteriorFullReparametrizedCorrelation(
                self.gaussian_prior.num_strains, self.gaussian_prior.num_times,
                dtype=self.dtype,
                initial_gaussian_bias=self.bias_initializer(self.gaussian_prior.population, self.gaussian_prior.times)
            )
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(self.correlation_type))

    # noinspection PyAttributeOutsideInit
    def precompile_elbo(self):
        """
        ELBO original formula:
            E_Q[P(X)] - E_Q[Q(X)] + E_{F ~ phi} [log P(F | X)]

        To obtain monte carlo estimates of ELBO, need to be able to compute:
        1. p(x)
        2. q(x), or more directly the entropy H(Q) = -E_Q[Q(X)]
        3. p(f|x) --> implemented via sparse log-likelihood matmul

        To save memory on larger inputs, split the ELBO up into several pieces.
        """
        logger.info("The first ELBO iteration will take longer due to JIT compilation.")
        n_data = np.array([
            len(self.data.time_slices[t_idx])
            for t_idx in range(self.gaussian_prior.num_times)
        ], dtype=self.dtype)  # length T
        n_times = self.gaussian_prior.num_times
        log_total_marker_lens = np.array([
            np.log(sum(len(m) for m in strain.markers))
            for strain in self.gaussian_prior.population.strains
        ], dtype=self.dtype)  # length S: total marker nucleotide length of each strain
        log_total_marker_lens = np.expand_dims(
            log_total_marker_lens, axis=[0, 1]
        )

        @jax.jit
        def _elbo(params, rand_samples):
            # entropic
            elbo = params['diag_weights'].sum()

            # model ll
            x = self.posterior.reparametrize(rand_samples, params)
            elbo += self.gaussian_prior.log_likelihood_x(x=x).mean()

            # read_frags ll
            log_y = jax.nn.log_softmax(x, axis=-1)
            for t_idx in range(n_times):  # this loop is OK to flatten via JIT
                # (1 x N x S)
                log_y_t = jax.lax.dynamic_slice_in_dim(log_y, start_index=t_idx, slice_size=1, axis=0).squeeze(0)
                n_singular = 0
                n_pairs = 0

                for batch_idx, batch_lls in enumerate(self.batches[t_idx]):
                    batch_sz = batch_lls.shape[1]
                    data_ll_part = batch_sz * log_mm_exp(log_y_t, batch_lls).mean()
                    elbo += data_ll_part
                    n_singular += batch_sz
                for paired_batch_idx, paired_batch_lls in enumerate(self.paired_batches[t_idx]):
                    batch_sz = paired_batch_lls.shape[1]
                    data_ll_part = batch_sz * log_mm_exp(log_y_t, paired_batch_lls).mean()
                    elbo += data_ll_part
                    n_pairs += batch_sz

                # Correction term: (log of) expected length of marker lens in population
                correction = -n_singular * jax.scipy.special.logsumexp(log_y_t + log_total_marker_lens, axis=-1).mean()  # mean across samples, one per read -> multiply by # of reads.
                elbo += correction

                # Correction term #2: (log of) expected length of square of marker lens in population
                correction = -n_pairs * jax.scipy.special.logsumexp(log_y_t + 2 * log_total_marker_lens, axis=-1).mean()
                elbo += correction

            # correction term
            correction = -n_data * jax.scipy.special.logsumexp(log_y + log_total_marker_lens, axis=-1).mean(axis=1)
            elbo += correction.sum()  # sum across timepoints
            return elbo
        self.elbo_grad = jax.value_and_grad(_elbo, argnums=0)

    # noinspection PyAttributeOutsideInit
    def precompile_elbo_pieces(self):
        assert isinstance(self.posterior, GaussianPosteriorFullReparametrizedCorrelation)
        # ================ JITted reparametrizations
        @jax.jit
        def _reparametrize_gaussian(params, rand_samples):
            return self.posterior.reparametrize(rand_samples, params)

        @jax.jit
        def _reparametrize_abundance(params, rand_samples):
            x = _reparametrize_gaussian(params, rand_samples)
            return jax.nn.log_softmax(x, axis=-1)

        @jax.jit
        def _reparametrize_abundance_t(params, rand_samples, t_idx):
            x = _reparametrize_gaussian(params, rand_samples)
            return jax.nn.log_softmax(
                jax.lax.dynamic_slice_in_dim(x, start_index=t_idx, slice_size=1, axis=0).squeeze(0),
                axis=-1
            )

        # ================== ELBO components
        @jax.jit
        def _elbo_entropy(params):
            return params['diag_weights'].sum()
        self.elbo_entropy = jax.value_and_grad(_elbo_entropy, argnums=0)

        @jax.jit
        def _elbo_model_prior(params, rand_samples):
            x = _reparametrize_gaussian(params, rand_samples)
            return self.gaussian_prior.log_likelihood_x(x=x).mean()
        self.elbo_model_prior = jax.value_and_grad(_elbo_model_prior, argnums=0)

        @jax.jit
        def _elbo_data_ll_t_batch(params, rand_samples, t_idx, batch_lls):
            batch_sz = batch_lls.shape[1]
            log_y_t = _reparametrize_abundance_t(params, rand_samples, t_idx)
            return batch_sz * log_mm_exp(log_y_t, batch_lls).mean()
        self.elbo_data_ll_t_batch = jax.value_and_grad(_elbo_data_ll_t_batch)

        @jax.jit
        def _elbo_data_correction(params, rand_samples, n_singular_data, n_paired_data,
                                  _log_total_marker_lens):
            log_y = _reparametrize_abundance(params, rand_samples)
            correction = -n_singular_data * jax.scipy.special.logsumexp(log_y + _log_total_marker_lens, axis=-1).mean(axis=1)  # mean across samples
            correction_paired = -n_paired_data * jax.scipy.special.logsumexp(log_y + (2 * _log_total_marker_lens), axis=-1).mean(axis=1)  # mean across samples
            return correction.sum() + correction_paired.sum()
        self.elbo_data_correction = jax.value_and_grad(_elbo_data_correction, argnums=0)

        self.n_data = np.expand_dims(
            np.array([
                sum(batch.shape[1] for batch in self.batches[t_idx])
                for t_idx in range(self.gaussian_prior.num_times)
            ], dtype=self.dtype),  # length T
            axis=1
        )
        self.n_paired_data = np.expand_dims(
            np.array([
                sum(batch.shape[1] for batch in self.paired_batches[t_idx])
                for t_idx in range(self.gaussian_prior.num_times)
            ], dtype=self.dtype),
            axis=1
        )
        log_total_marker_lens = np.array([
            cnp.log(sum(len(m) for m in strain.markers))
            for strain in self.gaussian_prior.population.strains
        ], dtype=self.dtype)  # length S: total marker nucleotide length of each strain
        self.log_total_marker_lens = np.expand_dims(log_total_marker_lens, axis=[0, 1])

    def elbo_with_grad(
            self,
            params: GENERIC_PARAM_TYPE,
            random_samples: GENERIC_SAMPLE_TYPE
    ) -> Tuple[np.ndarray, GENERIC_GRAD_TYPE]:
        if self.accumulate_gradients:
            acc_elbo_value, acc_elbo_grad = self.elbo_entropy(params)

            _e, _g = self.elbo_model_prior(params, random_samples)
            acc_elbo_value += _e
            acc_elbo_grad = accumulate_gradients(acc_elbo_grad, _g)

            for t_idx in range(self.gaussian_prior.num_times):
                for batch_lls in self.batches[t_idx]:
                    _e, _g = self.elbo_data_ll_t_batch(params, random_samples, t_idx, batch_lls)
                    acc_elbo_value += _e
                    acc_elbo_grad = accumulate_gradients(acc_elbo_grad, _g)

            _e, _g = self.elbo_data_correction(params, random_samples, self.n_data, self.n_paired_data, self.log_total_marker_lens)
            acc_elbo_value += _e
            acc_elbo_grad = accumulate_gradients(acc_elbo_grad, _g)
            return acc_elbo_value, acc_elbo_grad
        else:
            return self.elbo_grad(params, random_samples)


def accumulate_gradients(grad1: GENERIC_GRAD_TYPE, grad2: GENERIC_GRAD_TYPE):
    return {
        k: grad1[k] + grad2[k]
        for k in grad1.keys()
    }