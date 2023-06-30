from typing import Optional

import jax
import jax.numpy as np
import numpy as cnp

from ..base import AbstractReparametrizedPosterior, _GENERIC_SAMPLE_TYPE, _GENERIC_PARAM_TYPE
from .util import tril_linear_transform_with_bias, tril_linear_transform_no_bias

from chronostrain.config import cfg
from chronostrain.logging import create_logger
logger = create_logger(__name__)


INIT_SCALE = 1.0


class GaussianStrainCorrelatedWithGlobalZerosPosterior(AbstractReparametrizedPosterior):
    """
    A zero-model posterior, where there is a posterior indicator for each strain (to be applied across all timepoints).
    """

    def __init__(self, num_strains, num_times, dtype, initial_gaussian_bias: Optional[np.ndarray] = None):
        logger.info("Initializing Strain-correlated (time-factorized) posterior with Global Zeros")
        self.num_strains = num_strains
        self.num_times = num_times
        self.dtype = dtype

        self.parameters = {}
        for t_idx in range(self.num_times):  # gaussians are parametrized by block-banded precision matrix.
            self.parameters[f'tril_weights_{t_idx}'] = np.zeros(
                (self.num_strains * (self.num_strains - 1)) // 2,
                dtype=dtype
            )
            self.parameters[f'diag_weights_{t_idx}'] = np.full(
                self.num_strains, fill_value=cnp.log(INIT_SCALE),
                dtype=dtype
            )

        if initial_gaussian_bias is None:
            self.parameters['bias'] = np.zeros(
                (self.num_times, 1, self.num_strains),
                dtype=dtype
            )
        else:
            self.parameters['bias'] = np.expand_dims(initial_gaussian_bias, axis=1)
        self.parameters['gumbel_mean'] = np.zeros(
            (2, self.num_strains),
            dtype=dtype
        )

    def set_parameters(self, params: _GENERIC_PARAM_TYPE):
        self.parameters = params

    def get_parameters(self) -> _GENERIC_PARAM_TYPE:
        return self.parameters

    def random_sample(self, num_samples: int) -> _GENERIC_SAMPLE_TYPE:
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

    def reparametrized_gaussians(self, z: np.ndarray, params: _GENERIC_PARAM_TYPE) -> np.ndarray:
        gaussians = np.zeros(z.shape, dtype=z.dtype)
        z0 = jax.lax.dynamic_slice_in_dim(z, start_index=0, slice_size=1, axis=0).squeeze(0)  # N x S
        x0 = tril_linear_transform_no_bias(
            params['tril_weights_0'],
            np.exp(params['diag_weights_0']),
            z0
        )  # N x S
        gaussians = gaussians.at[0].set(x0)

        # z_prev = z0
        for t in range(1, self.num_times):
            z_t = jax.lax.dynamic_slice_in_dim(z, t, slice_size=1, axis=0).squeeze(0)  # N x S
            x_t = tril_linear_transform_no_bias(
                params[f'tril_weights_{t}'],
                np.exp(params[f'diag_weights_{t}']),
                z_t
            )
            # x_t = tril_linear_transform_no_bias(
            #     params[f'tril_weights_{t}'],
            #     np.exp(params[f'diag_weights_{t}']),
            #     z_t
            # ) + np.matmul(z_prev, params[f'cond_matrix_{t}'].T)
            # z_prev = z_t

            # x_t = tril_linear_transform_no_bias(
            #     params[f'tril_weights_{t}'],
            #     np.exp(params[f'diag_weights_{t}']),
            #     z_t
            # ) + np.matmul(x_prev, params[f'cond_matrix_{t}'].T)
            # x_t = tril_linear_transform_no_bias(
            #     params[f'tril_weights_{t}'],
            #     np.exp(params[f'diag_weights_{t}']),
            #     z_t + np.matmul(x_prev, params[f'cond_matrix_{t}'].T)
            # )
            # print("x{} shape: {}".format(t, x0.shape))
            gaussians = gaussians.at[t].set(x_t)
            # x_prev = x_t
        return gaussians + params['bias']

    def reparametrized_log_zeros_smooth(self, g: np.ndarray, params: _GENERIC_PARAM_TYPE, temp: float) -> np.ndarray:
        return jax.nn.log_softmax(
            (1 / temp) * (
                    g  # (2 x N x S)
                    +
                    np.expand_dims(params['gumbel_mean'], 1)  # (2 x S)
            ),
            axis=0
        )

    def reparametrized_zeros(self, g: np.ndarray, params: _GENERIC_PARAM_TYPE) -> np.ndarray:
        return np.less(
            g[0] + params['gumbel_mean'][0],
            g[1] + params['gumbel_mean'][1]
        )

    # noinspection PyMethodOverriding
    def reparametrize(self, random_samples: _GENERIC_SAMPLE_TYPE, params: _GENERIC_PARAM_TYPE, temp: float) -> _GENERIC_SAMPLE_TYPE:
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
    def entropy(self, params: _GENERIC_PARAM_TYPE, logits: np.ndarray, temp: float) -> np.ndarray:
        # Gaussian entropy
        ans = params[f'diag_weights_0'].sum()
        for t in range(1, self.num_times):
            ans += params[f'diag_weights_{t}'].sum()

        # bernoulli entropy
        gm = params['gumbel_mean']
        p = jax.nn.softmax(gm, axis=0)  # 2 x S, [p, 1-p] along axis 0
        logp = jax.nn.log_softmax(gm, axis=0)  # 2 x S, [log(p), log(1-p)] along axis 0
        ans += -np.sum(p * logp)

        # bernoulli entropy merely tries to keep the two mean parameters equal; regularize to prevent blowups.
        # Regularization here merely tries to keep MU_0 close to zero;
        # Note: the only determining factor in softmax is the gap (MU_1 - MU_0)
        ans += -np.square(jax.lax.dynamic_slice_in_dim(gm, start_index=0, slice_size=1, axis=0)).sum()

        # # concrete entropy, empirical
        # # logits: 2 x N x S
        # gm = params['gumbel_mean']  # 2 x S
        # ans += -gm.sum()
        # ans += (temp + 1) * logits.mean(axis=1).sum()
        # ans += 2 * jax.nn.logsumexp(   # logsumexp yields (N x S)
        #     np.expand_dims(gm, axis=1) - temp * logits,  # (2 x 1 x S) minus (2 x N x S)
        #     axis=0,
        #     keepdims=False
        # ).sum(axis=-1).mean()

        return ans


class GaussianTimeCorrelatedWithGlobalZerosPosterior(AbstractReparametrizedPosterior):
    """
    A zero-model posterior, where there is a posterior indicator for each strain (to be applied across all timepoints).
    """

    def __init__(self, num_strains, num_times, dtype, initial_gaussian_bias: Optional[np.ndarray] = None):
        logger.info("Initializing Time-correlated (strain-factorized) posterior with Global Zeros")
        self.num_strains = num_strains
        self.num_times = num_times
        self.dtype = dtype

        self.parameters = {}
        for s_idx in range(self.num_strains):  # gaussians are parametrized by block-banded precision matrix.
            self.parameters[f'tril_weights_{s_idx}'] = np.zeros(
                (self.num_times * (self.num_times - 1)) // 2,
                dtype=dtype
            )
            self.parameters[f'diag_weights_{s_idx}'] = np.full(
                self.num_times, fill_value=cnp.log(INIT_SCALE),
                dtype=dtype
            )

        if initial_gaussian_bias is None:
            self.parameters['bias'] = np.zeros(
                (self.num_times, 1, self.num_strains),
                dtype=dtype
            )
        else:
            self.parameters['bias'] = np.expand_dims(initial_gaussian_bias, axis=1)
        self.parameters['gumbel_mean'] = np.zeros(
            (2, self.num_strains),
            dtype=dtype
        )

    def set_parameters(self, params: _GENERIC_PARAM_TYPE):
        self.parameters = params

    def get_parameters(self) -> _GENERIC_PARAM_TYPE:
        return self.parameters

    def random_sample(self, num_samples: int) -> _GENERIC_SAMPLE_TYPE:
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

    def reparametrized_gaussians(self, z: np.ndarray, params: _GENERIC_PARAM_TYPE) -> np.ndarray:
        gaussians = np.zeros(z.shape, dtype=z.dtype)
        z0 = jax.lax.dynamic_slice_in_dim(z, start_index=0, slice_size=1, axis=0).squeeze(0)  # N x T
        x0 = tril_linear_transform_no_bias(
            params['tril_weights_0'],
            np.exp(params['diag_weights_0']),
            z0
        )  # N x T
        gaussians = gaussians.at[0].set(x0)

        for s in range(1, self.num_strains):
            z_s = jax.lax.dynamic_slice_in_dim(z, s, slice_size=1, axis=0).squeeze(0)  # N x T
            x_s = tril_linear_transform_no_bias(
                params[f'tril_weights_{s}'],
                np.exp(params[f'diag_weights_{s}']),
                z_s
            )
            gaussians = gaussians.at[s].set(x_s)
        return gaussians.transpose([2, 1, 0]) + params['bias']

    def reparametrized_log_zeros_smooth(self, g: np.ndarray, params: _GENERIC_PARAM_TYPE, temp: float) -> np.ndarray:
        return jax.nn.log_softmax(
            (1 / temp) * (
                    g  # (2 x N x S)
                    +
                    np.expand_dims(params['gumbel_mean'], 1)  # (2 x S)
            ),
            axis=0
        )

    def reparametrized_zeros(self, g: np.ndarray, params: _GENERIC_PARAM_TYPE) -> np.ndarray:
        return np.less(
            g[0] + params['gumbel_mean'][0],
            g[1] + params['gumbel_mean'][1]
        )

    # noinspection PyMethodOverriding
    def reparametrize(self, random_samples: _GENERIC_SAMPLE_TYPE, params: _GENERIC_PARAM_TYPE, temp: float) -> _GENERIC_SAMPLE_TYPE:
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
    def entropy(self, params: _GENERIC_PARAM_TYPE, logits: np.ndarray, temp: float) -> np.ndarray:
        # Gaussian entropy
        ans = params[f'diag_weights_0'].sum()
        for t in range(1, self.num_times):
            ans += params[f'diag_weights_{t}'].sum()

        # bernoulli entropy
        gm = params['gumbel_mean']
        p = jax.nn.softmax(gm, axis=0)  # 2 x S, [p, 1-p] along axis 0
        logp = jax.nn.log_softmax(gm, axis=0)  # 2 x S, [log(p), log(1-p)] along axis 0
        ans += -np.sum(p * logp)

        # bernoulli entropy merely tries to keep the two mean parameters equal; regularize to prevent blowups.
        # Regularization here merely tries to keep MU_0 close to zero;
        # Note: the only determining factor in softmax is the gap (MU_1 - MU_0)
        ans += -np.square(jax.lax.dynamic_slice_in_dim(gm, start_index=0, slice_size=1, axis=0)).sum()

        # # concrete entropy, empirical
        # # logits: 2 x N x S
        # gm = params['gumbel_mean']  # 2 x S
        # ans += -gm.sum()
        # ans += (temp + 1) * logits.mean(axis=1).sum()
        # ans += 2 * jax.nn.logsumexp(   # logsumexp yields (N x S)
        #     np.expand_dims(gm, axis=1) - temp * logits,  # (2 x 1 x S) minus (2 x N x S)
        #     axis=0,
        #     keepdims=False
        # ).sum(axis=-1).mean()

        return ans


class GaussianWithGlobalZerosPosteriorDense(AbstractReparametrizedPosterior):
    """
    A zero-model posterior, where there is a posterior indicator for each strain (to be applied across all timepoints).
    """

    def __init__(self, num_strains, num_times, dtype, initial_gaussian_bias: Optional[np.ndarray] = None):
        logger.info("Initializing Fully joint posterior with Global Zeros")
        self.num_strains = num_strains
        self.num_times = num_times

        self.parameters = {}

        n_features = self.num_times * self.num_strains
        self.parameters['tril_weights'] = np.zeros((n_features * (n_features - 1)) // 2, dtype=dtype)
        self.parameters['diag_weights'] = np.full(n_features, fill_value=cnp.log(INIT_SCALE), dtype=dtype)
        self.parameters['bias'] = np.zeros(n_features, dtype=dtype)
        self.parameters['gumbel_diff'] = np.zeros(self.num_strains, dtype=dtype)  # mu_0 - mu_1

        if initial_gaussian_bias is None:
            self.parameters['bias'] = np.zeros(n_features, dtype=dtype)
        else:
            self.parameters['bias'] = np.flatten(initial_gaussian_bias)  # Assumes shape is (n_times, n_strains)

    def set_parameters(self, params: _GENERIC_PARAM_TYPE):
        self.parameters = params

    def get_parameters(self) -> _GENERIC_PARAM_TYPE:
        return self.parameters

    def random_sample(self, num_samples: int) -> _GENERIC_SAMPLE_TYPE:
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

    def reparametrized_gaussians(self, z: np.ndarray, params: _GENERIC_PARAM_TYPE) -> np.ndarray:
        n_samples = z.shape[0]
        return tril_linear_transform_with_bias(
            params['tril_weights'],
            np.exp(params['diag_weights']),
            params['bias'],
            z
        ).reshape(n_samples, self.num_times, self.num_strains).transpose([1, 0, 2])

    def reparametrized_log_zeros_smooth(self, g: np.ndarray, params: _GENERIC_PARAM_TYPE, temp: float) -> np.ndarray:
        # compute the log-logistic sigmoid function LOG[ 1/(1+exp(-[g1-g2])) ].
        # g1 = mu_1 + G1
        # g2 = mu_2 + G2
        return jax.nn.log_softmax(
            (1 / temp) * g.at[0].add(params['gumbel_diff']),
            axis=0
        )

    def reparametrized_zeros(self, g: np.ndarray, params: _GENERIC_PARAM_TYPE) -> np.ndarray:
        return np.less(
            g[0] + params['gumbel_diff'],
            g[1]
        )

    # noinspection PyMethodOverriding
    def reparametrize(self, random_samples: _GENERIC_SAMPLE_TYPE, params: _GENERIC_PARAM_TYPE, temp: float) -> _GENERIC_SAMPLE_TYPE:
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
    def entropy(self, params: _GENERIC_PARAM_TYPE, logits: np.ndarray, temp: float) -> np.ndarray:
        # Gaussian entropy
        ans = params['diag_weights'].sum()
        # ans = np.array(0)
        # for t in range(self.num_times):
        #     ans += params[f'diag_weights_{t}'].sum()

        # bernoulli entropy
        g_diff = params['gumbel_diff']
        p = jax.scipy.special.expit(-g_diff)  # 1 / (1 + exp(delta_g = mu_0-mu_1)), note the minus sign!
        logp = -np.logaddexp(
            np.zeros(g_diff.shape, dtype=g_diff.dtype),
            g_diff
        )
        ans += -np.sum(p * logp)

        return ans
