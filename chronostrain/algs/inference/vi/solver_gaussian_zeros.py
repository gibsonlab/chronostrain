from typing import *

import jax
import jax.numpy as np
import numpy as cnp

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.optimization import LossOptimizer

from .base import AbstractADVISolver, _GENERIC_PARAM_TYPE, _GENERIC_SAMPLE_TYPE, _GENERIC_GRAD_TYPE, \
    AbstractReparametrizedPosterior
from .posteriors import GaussianStrainCorrelatedWithGlobalZerosPosterior, GaussianWithGlobalZerosPosteriorDense, GaussianTimeCorrelatedWithGlobalZerosPosterior

from chronostrain.logging import create_logger
from chronostrain.model.zeros.gumbel import PopulationGlobalZeros
from .util import log_mm_exp

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
                 optimizer: LossOptimizer,
                 prune_strains: bool,
                 read_batch_size: int = 5000,
                 accumulate_gradients: bool = False,
                 correlation_type: str = "full",
                 dtype='bfloat16',
                 initial_gaussian_bias: Optional[np.ndarray] = None):
        logger.info("Initializing solver with Gaussian-Zero posterior")
        self.dtype = dtype
        self.correlation_type = correlation_type
        self.initial_gaussian_bias = initial_gaussian_bias
        super().__init__(
            model=model,
            data=data,
            db=db,
            optimizer=optimizer,
            prune_strains=prune_strains,
            read_batch_size=read_batch_size
        )
        self.temperature = np.array(100, dtype=dtype)
        self.n_epochs_at_current_temp = 0
        self.zero_model = zero_model

        self.accumulate_gradients = accumulate_gradients
        if self.accumulate_gradients:
            self.precompile_elbo_pieces()
        else:
            self.precompile_elbo()
        logger.info("The first ELBO iteration will take longer due to JIT compilation.")

    def create_posterior(self) -> AbstractReparametrizedPosterior:
        if self.correlation_type == "full":
            return GaussianWithGlobalZerosPosteriorDense(
                self.model.num_strains(),
                self.model.num_times(),
                dtype=self.dtype, initial_gaussian_bias=self.initial_gaussian_bias
            )
        elif self.correlation_type == "time":
            return GaussianTimeCorrelatedWithGlobalZerosPosterior(
                self.model.num_strains(),
                self.model.num_times(),
                dtype=self.dtype,
                initial_gaussian_bias=self.initial_gaussian_bias
            )
        elif self.correlation_type == "strain":
            return GaussianStrainCorrelatedWithGlobalZerosPosterior(
                self.model.num_strains(),
                self.model.num_times(),
                dtype=self.dtype,
                initial_gaussian_bias=self.initial_gaussian_bias
            )
        else:
            raise ValueError("Unrecognized `correlation_type` argument {}.".format(self.correlation_type))

    def advance_epoch(self):
        if self.n_epochs_at_current_temp > 20:
            self.temperature = 0.5 * self.temperature
            self.n_epochs_at_current_temp = 0
        else:
            self.n_epochs_at_current_temp += 1

    def precompile_elbo(self):
        n_times = self.model.num_times()
        n_data = np.expand_dims(
            np.array([
                len(self.data.time_slices[t_idx])
                for t_idx in range(self.model.num_times())
            ], dtype=self.dtype),  # length T
            axis=1
        )
        log_total_marker_lens = np.array([
            cnp.log(sum(len(m) for m in strain.markers))
            for strain in self.model.bacteria_pop.strains
        ], dtype=self.dtype)  # length S: total marker nucleotide length of each strain

        @jax.jit
        def _elbo(params, rand_samples, temperature):
            reparam_samples = self.posterior.reparametrize(rand_samples, params, temperature)
            x = reparam_samples['gaussians']
            logits = reparam_samples['smooth_log_zeros']  # 2 x N x S

            # entropic
            elbo = self.posterior.entropy(params, logits, temperature)

            # model ll
            elbo += self.model.log_likelihood_x(x=x).mean()
            # elbo += self.zero_model.log_likelihood(logz).mean()  # Note that this is actually a constant.

            # data ll
            log_y = jax.nn.log_softmax(
                # (T x N x S) plus (1 x N x S)  ->  latter was obtained via slicing a 2 x N x S at the second "row".
                x + jax.lax.dynamic_slice_in_dim(logits, start_index=1, slice_size=1, axis=0),
                axis=-1
            )

            # both loops here are OK to flatten by JIT (data-dependent but fixed throughout inference)
            for t_idx in range(n_times):
                log_y_t = jax.lax.dynamic_slice_in_dim(log_y, start_index=t_idx, slice_size=1, axis=0).squeeze(0)

                for batch_idx, batch_lls in enumerate(self.batches[t_idx]):
                    batch_sz = batch_lls.shape[1]
                    data_ll_part = batch_sz * log_mm_exp(log_y_t, batch_lls).mean()
                    elbo += data_ll_part

            # correction term
            correction = -n_data * jax.scipy.special.logsumexp(log_y + log_total_marker_lens, axis=-1).mean(axis=1)  # mean across samples
            elbo += correction.sum()  # sum across timepoints
            return elbo
        self.elbo_grad = jax.value_and_grad(_elbo, argnums=0)

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

        @jax.jit
        def _elbo_model_prior(params, rand_samples):
            x = _reparametrize_gaussian(params, rand_samples)
            return self.model.log_likelihood_x(x=x).mean()
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
                for t_idx in range(self.model.num_times())
            ], dtype=self.dtype),  # length T
            axis=1
        )
        log_total_marker_lens = np.array([
            cnp.log(sum(len(m) for m in strain.markers))
            for strain in self.model.bacteria_pop.strains
        ], dtype=self.dtype)  # length S: total marker nucleotide length of each strain
        self.log_total_marker_lens = np.expand_dims(log_total_marker_lens, axis=[0, 1])

    def elbo_with_grad(self,
                       params: _GENERIC_PARAM_TYPE,
                       random_samples: _GENERIC_SAMPLE_TYPE
                       ) -> Tuple[np.ndarray, _GENERIC_GRAD_TYPE]:
        if self.accumulate_gradients:
            acc_elbo_value, acc_elbo_grad = self.elbo_entropy(params, random_samples, self.temperature)

            _e, _g = self.elbo_model_prior(params, random_samples)
            acc_elbo_value += _e
            acc_elbo_grad = accumulate_gradients(acc_elbo_grad, _g)

            for t_idx in range(self.model.num_times()):
                for batch_lls in self.batches[t_idx]:
                    _e, _g = self.elbo_data_ll_t_batch(params, random_samples, self.temperature, t_idx, batch_lls)
                    acc_elbo_value += _e
                    acc_elbo_grad = accumulate_gradients(acc_elbo_grad, _g)

            _e, _g = self.elbo_data_correction(params, random_samples, self.temperature, self.n_data, self.log_total_marker_lens)
            acc_elbo_value += _e
            acc_elbo_grad = accumulate_gradients(acc_elbo_grad, _g)
            return acc_elbo_value, acc_elbo_grad
        else:
            return self.elbo_grad(params, random_samples, self.temperature)


def accumulate_gradients(grad1: _GENERIC_GRAD_TYPE, grad2: _GENERIC_GRAD_TYPE):
    return {
        k: grad1[k] + grad2[k]
        for k in grad1.keys()
    }