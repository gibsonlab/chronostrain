from typing import *

import jax
import jax.numpy as np
import numpy as cnp

from chronostrain.model import Population, AbundanceGaussianPrior, AbstractErrorModel, \
    PopulationGlobalZeros, TimeSeriesReads
from chronostrain.database import StrainDatabase
from chronostrain.util.optimization import LossOptimizer

from chronostrain.inference.algs.vi.base.util import log_mm_exp
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
                 adhoc_corr_threshold: float = 0.99,
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
            read_batch_size=read_batch_size,
            adhoc_corr_threshold=adhoc_corr_threshold
        )
        self.temperature = np.array(10., dtype=dtype)
        self.temp_min = 1e-4
        self.n_epochs_at_current_temp = 0
        self.zero_model = zero_model

        self.log_total_marker_lens = np.array([
            cnp.log(sum(len(m) for m in strain.markers))
            for strain in self.gaussian_prior.population.strains
        ], dtype=self.dtype)  # length S: total marker nucleotide length of each strain

        self.accumulate_gradients = accumulate_gradients
        if self.accumulate_gradients:
            self.precompile_elbo_pieces()
        else:
            self.precompile_elbo()
        logger.info("The first ELBO iteration will take longer due to JIT compilation.")

    def create_posterior(self) -> GaussianWithGumbelsPosterior:
        if self.correlation_type == "full":
            logger.debug("Posterior is Mean-field q(X)q(Z) with full covariance matrix.")
            return GaussianWithGlobalZerosPosteriorDense(
                self.gaussian_prior.num_strains,
                self.gaussian_prior.num_times,
                dtype=self.dtype,
                initial_gaussian_bias=self.bias_initializer(self.gaussian_prior.population, self.gaussian_prior.times)
            )
        elif self.correlation_type == "time":
            logger.debug("Posterior is Mean-field q(X_1)...q(X_s)q(Z) split across strains.")
            return GaussianTimeCorrelatedWithGlobalZerosPosterior(
                self.gaussian_prior.num_strains,
                self.gaussian_prior.num_times,
                dtype=self.dtype,
                initial_gaussian_bias=self.bias_initializer(self.gaussian_prior.population, self.gaussian_prior.times)
            )
        elif self.correlation_type == "strain":
            logger.debug("Posterior is Mean-field q(X_1)...q(X_t)q(Z) split across time.")
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
        # ================ JITted reparametrizations
        assert isinstance(self.posterior, GaussianWithGumbelsPosterior)
        reparam_fn = jax.jit(self.posterior.reparametrize)
        entropy_fn = jax.jit(self.posterior.entropy)
        gaussian_ll_fn = jax.jit(self.gaussian_prior.log_likelihood_x)
        if self.zero_model.prior_p == 0.5:
            zero_ll_fn = jax.jit(lambda z: np.zeros(shape=(1,)))
        else:
            zero_ll_fn = jax.jit(self.zero_model.log_likelihood)

        @jax.jit
        def _elbo(params, rand_samples, temperature, batches, paired_batches, log_total_marker_lens):
            reparam_samples = reparam_fn(rand_samples, params, temperature)
            x = reparam_samples['gaussians']
            log_zeros = reparam_samples['smooth_log_zeros']  # 2 x N x S

            # entropic
            elbo = entropy_fn(params, log_zeros, temperature)

            # model ll
            elbo += gaussian_ll_fn(x=x).mean()
            elbo += zero_ll_fn(
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
            for t_idx in range(len(batches)):
                n_singular = 0
                n_pairs = 0
                log_y_t = jax.lax.dynamic_slice_in_dim(log_y, start_index=t_idx, slice_size=1, axis=0).squeeze(0)  # shape is (N x S)

                for batch_lls in batches[t_idx]:
                    batch_sz = batch_lls.shape[1]
                    data_ll_part = batch_sz * log_mm_exp(log_y_t, batch_lls).mean()
                    elbo += data_ll_part
                    n_singular += batch_sz
                for paired_batch_lls in paired_batches[t_idx]:
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
        assert isinstance(self.posterior, GaussianWithGumbelsPosterior)
        reparam_gaussians_fn = jax.jit(self.posterior.reparametrized_gaussians)
        reparam_logzeros_fn = jax.jit(self.posterior.reparametrized_log_zeros_smooth)
        entropy_fn = jax.jit(self.posterior.entropy)
        gaussian_ll_fn = jax.jit(self.gaussian_prior.log_likelihood_x)
        zero_ll_fn = jax.jit(self.zero_model.log_likelihood)

        @jax.jit
        def _reparametrize_gaussian(params, rand_samples):
            return reparam_gaussians_fn(rand_samples['std_gaussians'], params)

        @jax.jit
        def _reparametrize_logzeros(params, rand_samples, temperature):
            return reparam_logzeros_fn(rand_samples['std_gumbels'], params, temperature)

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
            return entropy_fn(params, logits, temperature)

        if self.zero_model.prior_p == 0.5:
            logger.debug("Precompiling with symmetric prior ELBO (p=0.5)")
            @jax.jit
            def _elbo_model_prior(params, rand_samples, temperature):
                x = _reparametrize_gaussian(params, rand_samples)
                return gaussian_ll_fn(x=x).mean()
        else:
            logger.debug("Precompiling with asymmetric prior ELBO (p={})".format(self.zero_model.prior_p))
            @jax.jit
            def _elbo_model_prior(params, rand_samples, temperature):
                x = _reparametrize_gaussian(params, rand_samples)
                gaussian_ll = gaussian_ll_fn(x=x).mean()
                log_zeros = _reparametrize_logzeros(params, rand_samples, temperature)
                return gaussian_ll + zero_ll_fn(
                    np.exp(
                        jax.lax.dynamic_slice_in_dim(log_zeros, start_index=1, slice_size=1, axis=0)
                    )
                ).mean()

        @jax.jit
        def _elbo_data_ll_t_batch(params, rand_samples, temperature, t_idx, batch_lls):
            batch_sz = batch_lls.shape[1]
            log_y_t = _reparametrize_abundance_t(params, rand_samples, temperature, t_idx)
            return batch_sz * log_mm_exp(log_y_t, batch_lls).mean()

        @jax.jit
        def _elbo_data_correction(params, rand_samples, temperature, n_singular_data, n_paired_data, _log_total_marker_lens):
            log_y = _reparametrize_abundance(params, rand_samples, temperature)
            correction = -n_singular_data * jax.scipy.special.logsumexp(log_y + _log_total_marker_lens, axis=-1).mean(axis=1)  # mean across samples
            correction_paired = -n_paired_data * jax.scipy.special.logsumexp(log_y + (2 * _log_total_marker_lens), axis=-1).mean(axis=1)  # mean across samples
            return correction.sum() + correction_paired.sum()

        self.elbo_entropy = jax.value_and_grad(_elbo_entropy, argnums=0)
        self.elbo_model_prior = jax.value_and_grad(_elbo_model_prior, argnums=0)
        self.elbo_data_ll_t_batch = jax.value_and_grad(_elbo_data_ll_t_batch)
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
        self.log_total_marker_lens = np.expand_dims(self.log_total_marker_lens, axis=[0, 1])

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
                for batch_lls in self.paired_batches[t_idx]:
                    _e, _g = self.elbo_data_ll_t_batch(params, random_samples, self.temperature, t_idx, batch_lls)
                    acc_elbo_value += _e
                    acc_elbo_grad = do_accumulate_gradients(acc_elbo_grad, _g)

            _e, _g = self.elbo_data_correction(params, random_samples, self.temperature,
                                               self.n_data, self.n_paired_data, self.log_total_marker_lens)
            acc_elbo_value += _e
            acc_elbo_grad = do_accumulate_gradients(acc_elbo_grad, _g)
            return acc_elbo_value, acc_elbo_grad
        else:
            return self.elbo_grad(
                params, random_samples, self.temperature,
                self.batches,
                self.paired_batches,
                self.log_total_marker_lens
            )


def do_accumulate_gradients(grad1: GENERIC_GRAD_TYPE, grad2: GENERIC_GRAD_TYPE):
    return {
        k: grad1[k] + grad2[k]
        for k in grad1.keys()
    }
