from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import jax
import jax.numpy as np
import numpy as cnp
from jax import Array

from ..base import AbstractReparametrizedPosterior, GENERIC_SAMPLE_TYPE, GENERIC_PARAM_TYPE
from .util import tril_linear_transform_with_bias, tril_linear_transform_no_bias

from chronostrain.config import cfg
from chronostrain.logging import create_logger
logger = create_logger(__name__)


INIT_SCALE = 1.0


class GaussianWithGumbelsPosterior(AbstractReparametrizedPosterior, ABC):
    @abstractmethod
    def reparametrized_gaussians(self, z: np.ndarray, params: GENERIC_PARAM_TYPE):
        """
        z: represents a standard (TxNxS) gaussian sample.
        """
        pass

    @abstractmethod
    def reparametrized_log_zeros_smooth(self, g: np.ndarray, params: GENERIC_PARAM_TYPE, temp: float) -> np.ndarray:
        """
        Returns smoothed (i.e. (0,1)-valued) booleans, where `temp` is the temperature parameter.
        High temperature means that the booleans are more smoothed out (since 1/temp approx. is 0)
        """
        pass

    @abstractmethod
    def reparametrized_zeros(self, g: np.ndarray, params: GENERIC_PARAM_TYPE) -> np.ndarray:
        """
        Returns binary {0,1} valued booleans.
        """
        pass


class GaussianStrainCorrelatedWithGlobalZerosPosterior(GaussianWithGumbelsPosterior):
    """
    A zero-model posterior, where there is a posterior indicator for each strain (to be applied across all timepoints).
    """
    def __init__(self, num_strains, num_times, dtype, initial_gaussian_bias: Optional[np.ndarray] = None):
        logger.info("Initializing Strain-correlated (time-factorized) posterior with Global Zeros")
        self.num_strains = num_strains
        self.num_times = num_times
        self.dtype = dtype
        self.initial_gaussian_bias = initial_gaussian_bias
        super().__init__()

    # noinspection PyTypeChecker
    def save_class_initializer(self, path: Path):
        with open(path, "wt") as f:
            print(".".join([self.__class__.__module__, self.__class__.__name__]), file=f)
            print(f"num_strains:int={self.num_strains}", file=f)
            print(f"num_times:int={self.num_times}", file=f)
            print(f"dtype:str={self.dtype}", file=f)

    def initial_params(self) -> GENERIC_PARAM_TYPE:
        parameters = {}
        for t_idx in range(self.num_times):  # gaussians are parametrized by block-banded precision matrix.
            parameters[f'tril_weights_{t_idx}'] = np.zeros(
                (self.num_strains * (self.num_strains - 1)) // 2,
                dtype=self.dtype
            )
            parameters[f'diag_weights_{t_idx}'] = np.full(
                self.num_strains, fill_value=cnp.log(INIT_SCALE),
                dtype=self.dtype
            )

        if self.initial_gaussian_bias is None:
            parameters['bias'] = np.zeros(
                (self.num_times, 1, self.num_strains),
                dtype=self.dtype
            )
        else:
            parameters['bias'] = np.expand_dims(self.initial_gaussian_bias, axis=1)
        parameters['gumbel_diff'] = np.zeros(self.num_strains, dtype=self.dtype)  # mu_0 - mu_1
        return parameters

    def set_parameters(self, params: GENERIC_PARAM_TYPE):
        self.parameters = params

    def get_parameters(self) -> GENERIC_PARAM_TYPE:
        return self.parameters

    def random_sample(self, num_samples: int) -> GENERIC_SAMPLE_TYPE:
        return {
            'std_gaussians': jax.random.normal(
                shape=[self.num_times, num_samples, self.num_strains],
                key=next(cfg.engine_cfg.generate_prng_keys(1)),
                dtype=self.dtype
            ),
            'std_gumbels': jax.random.gumbel(
                shape=[2, num_samples, self.num_strains],
                key=next(cfg.engine_cfg.generate_prng_keys(1)),
                dtype=self.dtype
            )
        }

    def reparametrized_gaussians(self, z: np.ndarray, params: GENERIC_PARAM_TYPE) -> np.ndarray:
        """
        z: represents a standard (TxNxS) gaussian sample.
        """
        gaussians: Array = np.zeros(z.shape, dtype=z.dtype)
        for t in range(self.num_times):
            z_t = jax.lax.dynamic_slice_in_dim(z, start_index=t, slice_size=1, axis=0).squeeze(0)  # N x S
            x_t = tril_linear_transform_no_bias(
                params[f'tril_weights_{t}'],
                np.exp(params[f'diag_weights_{t}']),
                z_t
            )
            gaussians = gaussians.at[t].set(x_t)
        return gaussians + params['bias']

    def reparametrized_log_zeros_smooth(self, g: np.ndarray, params: GENERIC_PARAM_TYPE, temp: float) -> np.ndarray:
        """
        Returns smoothed (i.e. (0,1)-valued) booleans, where `temp` is the temperature parameter.
        High temperature means that the booleans are more smoothed out (since 1/temp approx. is 0)
        """
        return jax.nn.log_softmax(
            (1 / temp) * g.at[0].add(params['gumbel_diff']),
            axis=0
        )

    def reparametrized_zeros(self, g: np.ndarray, params: GENERIC_PARAM_TYPE) -> np.ndarray:
        """
        Returns binary {0,1} valued booleans.
        """
        return np.less(
            g[0] + params['gumbel_diff'],
            g[1]
        )

    # noinspection PyMethodOverriding
    def reparametrize(self, random_samples: GENERIC_SAMPLE_TYPE, params: GENERIC_PARAM_TYPE, temp: float) -> GENERIC_SAMPLE_TYPE:
        return {
            'gaussians': self.reparametrized_gaussians(random_samples['std_gaussians'], params),
            'smooth_log_zeros': self.reparametrized_log_zeros_smooth(random_samples['std_gumbels'], params, temp)
        }

    def abundance_sample(self, num_samples: int = 1) -> np.ndarray:
        rand = self.random_sample(num_samples)
        g = self.reparametrized_gaussians(rand['std_gaussians'], self.get_parameters())  # T x N x S
        z = self.reparametrized_zeros(rand['std_gumbels'], self.get_parameters())  # N x S
        return jax.nn.softmax(g + np.expand_dims(np.log(z), axis=0), axis=-1)

    # noinspection PyMethodOverriding
    def entropy(self, params: GENERIC_PARAM_TYPE, logits: np.ndarray, temp: float) -> np.ndarray:
        # Gaussian entropy
        ans = params[f'diag_weights_0'].sum()
        for t in range(1, self.num_times):
            ans += params[f'diag_weights_{t}'].sum()

        # bernoulli entropy
        g_diff = params['gumbel_diff']
        p = jax.scipy.special.expit(-g_diff)  # 1 / (1 + exp(delta_g = mu_0-mu_1)), note the minus sign!
        logp = -np.logaddexp(
            np.zeros(g_diff.shape, dtype=g_diff.dtype),
            g_diff
        )
        ans += -np.sum(p * logp)
        return ans


class GaussianTimeCorrelatedWithGlobalZerosPosterior(GaussianWithGumbelsPosterior):
    """
    A zero-model posterior, where there is a posterior indicator for each strain (to be applied across all timepoints).
    """

    def __init__(self, num_strains, num_times, dtype, initial_gaussian_bias: Optional[np.ndarray] = None):
        logger.info("Initializing Time-correlated (strain-factorized) posterior with Global Zeros")
        self.num_strains = num_strains
        self.num_times = num_times
        self.dtype = dtype
        self.initial_gaussian_bias = initial_gaussian_bias
        super().__init__()

    # noinspection PyTypeChecker
    def save_class_initializer(self, path: Path):
        with open(path, "wt") as f:
            print(".".join([self.__class__.__module__, self.__class__.__name__]), file=f)
            print(f"num_strains:int={self.num_strains}", file=f)
            print(f"num_times:int={self.num_times}", file=f)
            print(f"dtype:str={self.dtype}", file=f)

    def initial_params(self) -> GENERIC_PARAM_TYPE:
        parameters = {}
        for s_idx in range(self.num_strains):  # gaussians are parametrized by block-banded precision matrix.
            parameters[f'tril_weights_{s_idx}'] = np.zeros(
                (self.num_times * (self.num_times - 1)) // 2,
                dtype=self.dtype
            )
            parameters[f'diag_weights_{s_idx}'] = np.full(
                self.num_times, fill_value=cnp.log(INIT_SCALE),
                dtype=self.dtype
            )

        if self.initial_gaussian_bias is None:
            parameters['bias'] = np.zeros(
                (self.num_times, 1, self.num_strains),
                dtype=self.dtype
            )
        else:
            parameters['bias'] = np.expand_dims(self.initial_gaussian_bias, axis=1)
        parameters['gumbel_diff'] = np.zeros(self.num_strains, dtype=self.dtype)  # mu_0 - mu_1
        return parameters

    def set_parameters(self, params: GENERIC_PARAM_TYPE):
        self.parameters = params

    def get_parameters(self) -> GENERIC_PARAM_TYPE:
        return self.parameters

    def random_sample(self, num_samples: int) -> GENERIC_SAMPLE_TYPE:
        return {
            'std_gaussians': jax.random.normal(
                shape=[self.num_strains, num_samples, self.num_times],
                key=next(cfg.engine_cfg.generate_prng_keys(1)),
                dtype=self.dtype
            ),
            'std_gumbels': jax.random.gumbel(
                shape=[2, num_samples, self.num_strains],
                key=next(cfg.engine_cfg.generate_prng_keys(1)),
                dtype=self.dtype
            )
        }

    def reparametrized_gaussians(self, z: np.ndarray, params: GENERIC_PARAM_TYPE) -> np.ndarray:
        gaussians: Array = np.zeros(z.shape, dtype=z.dtype)
        for s in range(1, self.num_strains):
            z_s = jax.lax.dynamic_slice_in_dim(z, start_index=s, slice_size=1, axis=0).squeeze(0)  # N x T
            x_s = tril_linear_transform_no_bias(
                params[f'tril_weights_{s}'],
                np.exp(params[f'diag_weights_{s}']),
                z_s
            )
            gaussians = gaussians.at[s].set(x_s)
        return gaussians.transpose([2, 1, 0]) + params['bias']

    def reparametrized_log_zeros_smooth(self, g: np.ndarray, params: GENERIC_PARAM_TYPE, temp: float) -> np.ndarray:
        # compute the log-logistic sigmoid function LOG[ 1/(1+exp(-[g1-g2])) ].
        # g1 = mu_1 + G1
        # g2 = mu_2 + G2
        return jax.nn.log_softmax(
            (1 / temp) * g.at[0].add(params['gumbel_diff']),
            axis=0
        )

    def reparametrized_zeros(self, g: np.ndarray, params: GENERIC_PARAM_TYPE) -> np.ndarray:
        return np.less(
            g[0] + params['gumbel_diff'],
            g[1]
        )

    # noinspection PyMethodOverriding
    def reparametrize(self, random_samples: GENERIC_SAMPLE_TYPE, params: GENERIC_PARAM_TYPE, temp: float) -> GENERIC_SAMPLE_TYPE:
        return {
            'gaussians': self.reparametrized_gaussians(random_samples['std_gaussians'], params),
            'smooth_log_zeros': self.reparametrized_log_zeros_smooth(random_samples['std_gumbels'], params, temp)
        }

    def abundance_sample(self, num_samples: int = 1) -> np.ndarray:
        rand = self.random_sample(num_samples)
        g = self.reparametrized_gaussians(rand['std_gaussians'], self.get_parameters())  # T x N x S
        z = self.reparametrized_zeros(rand['std_gumbels'], self.get_parameters())  # N x S
        return jax.nn.softmax(g + np.expand_dims(np.log(z), axis=0), axis=-1)

    # noinspection PyMethodOverriding
    def entropy(self, params: GENERIC_PARAM_TYPE, logits: np.ndarray, temp: float) -> np.ndarray:
        # Gaussian entropy
        ans = params[f'diag_weights_0'].sum()
        for s in range(1, self.num_strains):
            ans += params[f'diag_weights_{s}'].sum()

        # bernoulli entropy
        g_diff = params['gumbel_diff']
        p = jax.scipy.special.expit(-g_diff)  # 1 / (1 + exp(delta_g = mu_0-mu_1)), note the minus sign!
        logp = -np.logaddexp(
            np.zeros(g_diff.shape, dtype=g_diff.dtype),
            g_diff
        )
        ans += -np.sum(p * logp)
        return ans


class GaussianWithGlobalZerosPosteriorDense(GaussianWithGumbelsPosterior):
    """
    A zero-model posterior, where there is a posterior indicator for each strain (to be applied across all timepoints).
    """

    def __init__(self, num_strains, num_times, dtype, initial_gaussian_bias: Optional[np.ndarray] = None):
        logger.info("Initializing Fully joint posterior with Global Zeros")
        self.num_strains = num_strains
        self.num_times = num_times
        self.dtype = dtype
        self.initial_gaussian_bias = initial_gaussian_bias
        super().__init__()

    # noinspection PyTypeChecker
    def save_class_initializer(self, path: Path):
        with open(path, "wt") as f:
            print(".".join([self.__class__.__module__, self.__class__.__name__]), file=f)
            print(f"num_strains:int={self.num_strains}", file=f)
            print(f"num_times:int={self.num_times}", file=f)
            print(f"dtype:str={self.dtype}", file=f)

    def initial_params(self) -> GENERIC_PARAM_TYPE:
        parameters = {}
        n_features = self.num_times * self.num_strains
        parameters['tril_weights'] = np.zeros((n_features * (n_features - 1)) // 2, dtype=self.dtype)
        parameters['diag_weights'] = np.full(n_features, fill_value=cnp.log(INIT_SCALE), dtype=self.dtype)
        parameters['bias'] = np.zeros(n_features, dtype=self.dtype)
        parameters['gumbel_diff'] = np.zeros(self.num_strains, dtype=self.dtype)  # mu_0 - mu_1

        if self.initial_gaussian_bias is None:
            parameters['bias'] = np.zeros(n_features, dtype=self.dtype)
        else:
            parameters['bias'] = np.ravel(self.initial_gaussian_bias)  # Assumes shape is (n_times, n_strains)
        return parameters

    def set_parameters(self, params: GENERIC_PARAM_TYPE):
        self.parameters = params

    def get_parameters(self) -> GENERIC_PARAM_TYPE:
        return self.parameters

    def random_sample(self, num_samples: int) -> GENERIC_SAMPLE_TYPE:
        return {
            'std_gaussians': jax.random.normal(
                shape=[num_samples, self.num_times * self.num_strains],
                key=next(cfg.engine_cfg.generate_prng_keys(num_keys=1))
            ),
            'std_gumbels': jax.random.gumbel(
                shape=[2, num_samples, self.num_strains],
                key=next(cfg.engine_cfg.generate_prng_keys(1))
            )
        }

    def reparametrized_gaussians(self, z: np.ndarray, params: GENERIC_PARAM_TYPE) -> np.ndarray:
        n_samples = z.shape[0]
        return tril_linear_transform_with_bias(
            params['tril_weights'],
            np.exp(params['diag_weights']),
            params['bias'],
            z
        ).reshape(n_samples, self.num_times, self.num_strains).transpose([1, 0, 2])

    def reparametrized_log_zeros_smooth(self, g: np.ndarray, params: GENERIC_PARAM_TYPE, temp: float) -> np.ndarray:
        # compute the log-logistic sigmoid function LOG[ 1/(1+exp(-[g1-g2])) ].
        # g1 = mu_1 + G1
        # g2 = mu_2 + G2
        return jax.nn.log_softmax(
            (1 / temp) * g.at[0].add(params['gumbel_diff']),
            axis=0
        )

    def reparametrized_zeros(self, g: np.ndarray, params: GENERIC_PARAM_TYPE) -> np.ndarray:
        return np.less(
            g[0] + params['gumbel_diff'],
            g[1]
        )

    # noinspection PyMethodOverriding
    def reparametrize(self, random_samples: GENERIC_SAMPLE_TYPE, params: GENERIC_PARAM_TYPE, temp: float) -> GENERIC_SAMPLE_TYPE:
        return {
            'gaussians': self.reparametrized_gaussians(random_samples['std_gaussians'], params),
            'smooth_log_zeros': self.reparametrized_log_zeros_smooth(random_samples['std_gumbels'], params, temp)
        }

    def abundance_sample(self, num_samples: int = 1) -> np.ndarray:
        rand = self.random_sample(num_samples)
        g = self.reparametrized_gaussians(rand['std_gaussians'], self.get_parameters())  # T x N x S
        z = self.reparametrized_zeros(rand['std_gumbels'], self.get_parameters())  # N x S
        return jax.nn.softmax(g + np.expand_dims(np.log(z), axis=0), axis=-1)

    # noinspection PyMethodOverriding
    def entropy(self, params: GENERIC_PARAM_TYPE, logits: np.ndarray, temp: float) -> np.ndarray:
        # Gaussian entropy
        ans = params['diag_weights'].sum()

        # bernoulli entropy
        g_diff = params['gumbel_diff']
        p = jax.scipy.special.expit(-g_diff)  # 1 / (1 + exp(delta_g = mu_0-mu_1)), note the minus sign!
        logp = -np.logaddexp(
            np.zeros(g_diff.shape, dtype=g_diff.dtype),
            g_diff
        )
        ans += -np.sum(p * logp)
        return ans
