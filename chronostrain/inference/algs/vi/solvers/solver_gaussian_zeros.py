import itertools
from typing import *

import jax
import jax.numpy as np
import numpy as cnp

from chronostrain.model import Population, AbundanceGaussianPrior, AbstractErrorModel, \
    PopulationGlobalZeros, TimeSeriesReads
from chronostrain.database import StrainDatabase
from chronostrain.util.optimization import LossOptimizer

from chronostrain.inference.algs.vi.base.util import log_mm_exp, log_mv_exp
from chronostrain.inference.algs.vi.base.advi import AbstractADVISolver
from ..posteriors import *
from ..base import GENERIC_GRAD_TYPE, GENERIC_PARAM_TYPE, GENERIC_SAMPLE_TYPE

from chronostrain.logging import create_logger
logger = create_logger(__name__)


# typedef
class BiasInitializer(Protocol):
    def __call__(self, population: Population, times: List[float]) -> np.ndarray:
        pass


class ADVIGaussianZerosSolver(AbstractADVISolver):
    """
    A basic implementation of ADVI estimating the posterior p(X|R), with fragments
    F marginalized out.
    """

    def __init__(self,
                 gaussian_prior: AbundanceGaussianPrior,
                 error_model: AbstractErrorModel,
                 zero_model: PopulationGlobalZeros,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 optimizer: LossOptimizer,
                 prune_strains: bool,
                 bias_initializer: BiasInitializer,
                 read_batch_size: int = 5000,
                 accumulate_gradients: bool = False,
                 correlation_type: str = "full",
                 dtype='bfloat16'):
        logger.info("Initializing solver with Gaussian-Zero posterior")
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
        self.temperature = np.array(10., dtype=dtype)
        self.temp_min = 1e-4
        self.n_epochs_at_current_temp = 0
        self.zero_model = zero_model

        self.accumulate_gradients = accumulate_gradients
        if self.accumulate_gradients:
            self.precompile_elbo_pieces()
        else:
            self.precompile_elbo()
        logger.info("The first ELBO iteration will take longer due to JIT compilation.")

    def create_posterior(self) -> GaussianWithGumbelsPosterior:
        if self.correlation_type == "full":
            return GaussianWithGlobalZerosPosteriorDense(
                self.gaussian_prior.num_strains,
                self.gaussian_prior.num_times,
                dtype=self.dtype,
                initial_gaussian_bias=self.bias_initializer(self.gaussian_prior.population, self.gaussian_prior.times)
            )
        elif self.correlation_type == "time":
            return GaussianTimeCorrelatedWithGlobalZerosPosterior(
                self.gaussian_prior.num_strains,
                self.gaussian_prior.num_times,
                dtype=self.dtype,
                initial_gaussian_bias=self.bias_initializer(self.gaussian_prior.population, self.gaussian_prior.times)
            )
        elif self.correlation_type == "strain":
            return GaussianStrainCorrelatedWithGlobalZerosPosterior(
                self.gaussian_prior.num_strains,
                self.gaussian_prior.num_times,
                dtype=self.dtype,
                initial_gaussian_bias=self.bias_initializer(self.gaussian_prior.population, self.gaussian_prior.times)
            )
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(self.correlation_type))

    def advance_epoch(self, epoch):
        __anneal_rate = 0.95
        self.temperature = np.maximum(self.temperature * __anneal_rate, self.temp_min)
        if epoch % 10 == 0:
            logger.debug("Temperature = {}".format(self.temperature))

    def precompile_elbo(self):
        n_times = self.gaussian_prior.num_times
        log_total_marker_lens = np.array([
            cnp.log(sum(len(m) for m in strain.markers))
            for strain in self.gaussian_prior.population.strains
        ], dtype=self.dtype)  # length S: total marker nucleotide length of each strain

        @jax.jit
        def _elbo(params, rand_samples, temperature):
            reparam_samples = self.posterior.reparametrize(rand_samples, params, temperature)
            x = reparam_samples['gaussians']
            log_zeros = reparam_samples['smooth_log_zeros']  # 2 x N x S

            # entropic
            elbo = self.posterior.entropy(params, log_zeros, temperature)

            # model ll
            elbo += self.gaussian_prior.log_likelihood_x(x=x).mean()
            if self.zero_model.prior_p != 0.5:
                elbo += self.zero_model.log_likelihood(
                    np.exp(
                        jax.lax.dynamic_slice_in_dim(log_zeros, start_index=1, slice_size=1, axis=0)
                    )
                ).mean()

            # read_frags ll
            log_y = jax.nn.log_softmax(
                # (T x N x S) plus (1 x N x S)  ->  latter was obtained via slicing a 2 x N x S at the second "row".
                x + jax.lax.dynamic_slice_in_dim(log_zeros, start_index=1, slice_size=1, axis=0),
                axis=-1
            )

            # both loops here are OK to flatten by JIT (read_frags-dependent but fixed throughout algs)
            for t_idx in range(n_times):
                n_singular = 0
                n_pairs = 0
                log_y_t = jax.lax.dynamic_slice_in_dim(log_y, start_index=t_idx, slice_size=1, axis=0).squeeze(0)  # shape is (N x S)

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
            return elbo
        self.elbo_grad = jax.value_and_grad(_elbo, argnums=0)

    # noinspection PyAttributeOutsideInit
    def precompile_elbo_pieces(self):
        # ================ JITted reparametrizations
        @jax.jit
        def _reparametrize_gaussian(params, rand_samples):
            return self.posterior.reparametrized_gaussians(
                rand_samples['std_gaussians'],
                params
            )

        @jax.jit
        def _reparametrize_logzeros(params, rand_samples, temperature):
            return self.posterior.reparametrized_log_zeros_smooth(
                rand_samples['std_gumbels'],
                params,
                temperature
            )

        @jax.jit
        def _reparametrize_abundance(params, rand_samples, temperature):
            x = _reparametrize_gaussian(params, rand_samples)
            logits = _reparametrize_logzeros(params, rand_samples, temperature)
            return jax.nn.log_softmax(
                # (T x N x S) plus (1 x N x S)  ->  latter was obtained via slicing a 2 x N x S at the second "row".
                x + jax.lax.dynamic_slice_in_dim(logits, start_index=1, slice_size=1, axis=0),
                axis=-1
            )

        @jax.jit
        def _reparametrize_abundance_t(params, rand_samples, temperature, t_idx):
            x = _reparametrize_gaussian(params, rand_samples)
            logits = _reparametrize_logzeros(params, rand_samples, temperature)
            return jax.nn.log_softmax(
                jax.lax.dynamic_slice_in_dim(x, start_index=t_idx, slice_size=1, axis=0).squeeze(0)
                +
                jax.lax.dynamic_slice_in_dim(logits, start_index=1, slice_size=1, axis=0).squeeze(0),
                axis=-1
            )

        # ================== ELBO components
        @jax.jit
        def _elbo_entropy(params, rand_samples, temperature):
            logits = _reparametrize_logzeros(params, rand_samples, temperature)
            return self.posterior.entropy(params, logits, temperature)
        self.elbo_entropy = jax.value_and_grad(_elbo_entropy, argnums=0)

        if self.zero_model.prior_p == 0.5:
            logger.debug("Precompiling with symmetric prior ELBO (p=0.5)")
            @jax.jit
            def _elbo_model_prior(params, rand_samples, temperature):
                x = _reparametrize_gaussian(params, rand_samples)
                return self.gaussian_prior.log_likelihood_x(x=x).mean()
        else:
            logger.debug("Precompiling with asymmetric prior ELBO (p={})".format(self.zero_model.prior_p))
            @jax.jit
            def _elbo_model_prior(params, rand_samples, temperature):
                x = _reparametrize_gaussian(params, rand_samples)
                gaussian_ll = self.gaussian_prior.log_likelihood_x(x=x).mean()
                log_zeros = _reparametrize_logzeros(params, rand_samples, temperature)
                return gaussian_ll + self.zero_model.log_likelihood(
                    np.exp(
                        jax.lax.dynamic_slice_in_dim(log_zeros, start_index=1, slice_size=1, axis=0)
                    )
                ).mean()
        self.elbo_model_prior = jax.value_and_grad(_elbo_model_prior, argnums=0)

        @jax.jit
        def _elbo_data_ll_t_batch(params, rand_samples, temperature, t_idx, batch_lls):
            batch_sz = batch_lls.shape[1]
            log_y_t = _reparametrize_abundance_t(params, rand_samples, temperature, t_idx)
            return batch_sz * log_mm_exp(log_y_t, batch_lls).mean()
        self.elbo_data_ll_t_batch = jax.value_and_grad(_elbo_data_ll_t_batch)

        @jax.jit
        def _elbo_data_correction(params, rand_samples, temperature, n_data, _log_total_marker_lens):
            log_y = _reparametrize_abundance(params, rand_samples, temperature)
            correction = -n_data * jax.scipy.special.logsumexp(log_y + _log_total_marker_lens, axis=-1).mean(axis=1)  # mean across samples
            return correction.sum()
        self.elbo_data_correction = jax.value_and_grad(_elbo_data_correction, argnums=0)

        self.n_data = np.expand_dims(
            np.array([
                len(self.data.time_slices[t_idx])
                for t_idx in range(self.gaussian_prior.num_times)
            ], dtype=self.dtype),  # length T
            axis=1
        )
        log_total_marker_lens = np.array([
            cnp.log(sum(len(m) for m in strain.markers))
            for strain in self.gaussian_prior.population.strains
        ], dtype=self.dtype)  # length S: total marker nucleotide length of each strain
        self.log_total_marker_lens = np.expand_dims(log_total_marker_lens, axis=[0, 1])

    def elbo_with_grad(self,
                       params: GENERIC_PARAM_TYPE,
                       random_samples: GENERIC_SAMPLE_TYPE
                       ) -> Tuple[np.ndarray, GENERIC_GRAD_TYPE]:
        if self.accumulate_gradients:
            acc_elbo_value, acc_elbo_grad = self.elbo_entropy(params, random_samples, self.temperature)

            _e, _g = self.elbo_model_prior(params, random_samples, self.temperature)
            acc_elbo_value += _e
            acc_elbo_grad = do_accumulate_gradients(acc_elbo_grad, _g)

            for t_idx in range(self.gaussian_prior.num_times):
                for batch_lls in self.batches[t_idx]:
                    _e, _g = self.elbo_data_ll_t_batch(params, random_samples, self.temperature, t_idx, batch_lls)
                    acc_elbo_value += _e
                    acc_elbo_grad = do_accumulate_gradients(acc_elbo_grad, _g)

            _e, _g = self.elbo_data_correction(params, random_samples, self.temperature, self.n_data, self.log_total_marker_lens)
            acc_elbo_value += _e
            acc_elbo_grad = do_accumulate_gradients(acc_elbo_grad, _g)
            return acc_elbo_value, acc_elbo_grad
        else:
            return self.elbo_grad(params, random_samples, self.temperature)


def do_accumulate_gradients(grad1: GENERIC_GRAD_TYPE, grad2: GENERIC_GRAD_TYPE):
    return {
        k: grad1[k] + grad2[k]
        for k in grad1.keys()
    }


# ============== JAX-specific optimizations.
def log_mm_exp2(x, y):
    return jax.lax.map(
        lambda _y: log_mv_exp(x, _y),
        y.T
    ).T

def recursive_checkpoint(funs):
    """From JAX documentation page of jax.checkpoint() implementing a binary recursive strategy."""
    if len(funs) == 1:
        return jax.checkpoint(funs[0])
    else:
        f1 = recursive_checkpoint(funs[:len(funs)//2])
        f2 = recursive_checkpoint(funs[len(funs)//2:])
        f12 = lambda x: f1(f2(x))
        return jax.checkpoint(f12)


def timepoint_recursive_checkpoint_fn(n_batches: int) -> Callable:
    """
    Uses the binary recursion to implement memory-efficient version of
    f(batches) = log_p(batch_1) + log_p(batch_2) + ... + log_p(batch_N).
    """
    if n_batches == 0:
        raise ValueError("Timepoint must contain at least one batch.")

    return recursive_checkpoint([
        aggregate_timepoint_data_ll_checkpoint
        for _ in range(n_batches)
    ])


def aggregate_timepoint_data_ll_checkpoint(
        args: Tuple[np.ndarray, List[np.ndarray], np.ndarray]
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    log_y_t, batch_lls, elbo_acc = args  # Unpack args.

    # Unwraps a single batch, and adds it to the elbo. Serves as a helper for the recursive strategy.
    return log_y_t, batch_lls[1:], elbo_acc + aggregate_single_batch(log_y_t, batch_lls[0])


def aggregate_single_batch(log_y_t: np.ndarray, batch_ll: np.ndarray):
    batch_sz = batch_ll.shape[1]
    return batch_sz * log_mm_exp(log_y_t, batch_ll).mean()
