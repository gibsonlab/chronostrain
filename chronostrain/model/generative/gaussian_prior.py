"""
 gaussian_prior.py
 Contains classes for representing the generative model.
"""
from typing import List

import jax
import jax.numpy as np
from chronostrain.config import cfg
from chronostrain.logging import create_logger
from chronostrain.model import Population

logger = create_logger(__name__)


class AbundanceGaussianPrior:
    def __init__(
            self,
            times: List[float],
            tau_1_dof: float,
            tau_1_scale: float,
            tau_dof: float,
            tau_scale: float,
            population: Population
    ):
        """
        :param times: A list of time points.
        :param tau_1_dof: The scale-inverse-chi-squared DOF of the first time point.
        :param tau_1_scale: The scale-inverse-chi-squared scale of the first time point.
        :param tau_dof: The scale-inverse-chi-squared DOF of the rest of the Gaussian process.
        :param tau_scale: The scale-inverse-chi-squared scale of the rest of the Gaussian process.
        """

        self.times: List[float] = times  # array of time points
        self.tau_1_dof: float = tau_1_dof
        self.tau_1_scale: float = tau_1_scale
        self.tau_dof: float = tau_dof
        self.tau_scale: float = tau_scale
        self._frag_freqs_sparse = None
        self._frag_freqs_dense = None
        self.population = population

        logger.debug(f"Model has inverse temperature = {cfg.model_cfg.inverse_temperature}")
        self.latent_conversion = lambda x: jax.nn.softmax(cfg.model_cfg.inverse_temperature * x, axis=-1)

        # self.log_latent_conversion = lambda x: torch.log(sparsemax(x, dim=-1))
        self.log_latent_conversion = lambda x: jax.nn.log_softmax(cfg.model_cfg.inverse_temperature * x, axis=-1)

        self.dt_sqrt_inverse = np.power(np.array(
            [
                self.dt(t_idx)
                for t_idx in range(1, self.num_times)
            ],
            dtype=cfg.engine_cfg.dtype
        ), -0.5)

    @property
    def num_times(self) -> int:
        return len(self.times)

    @property
    def num_strains(self) -> int:
        return len(self.population)

    def log_likelihood_x(self, x: np.ndarray) -> np.ndarray:
        """
        Given an (T x N x S) tensor where N = # of instances/samples of X, compute the N different log-likelihoods.
        """
        if len(x.shape) == 2:
            r, c = x.shape
            x = x.reshape(r, 1, c)
        return self.log_likelihood_x_jeffreys_prior(x)

    def log_likelihood_x_jeffreys_prior(self, x: np.ndarray) -> np.ndarray:
        """

        Implementation of log_likelihood_x using Jeffrey's prior (for the Gaussian with known mean) for the variance.
        Assumes that the shape of X is constant (and only returns the non-constant part.)
        """
        n_times, _, n_strains = x.shape

        ll_first = -0.5 * n_strains * np.log(np.square(
            x[0, :, :]
        ).sum(axis=-1))

        if n_times > 1:
            ll_rest = -0.5 * (n_times - 1) * n_strains * np.log(np.square(
                np.expand_dims(self.dt_sqrt_inverse, axis=[1, 2]) * np.diff(x, n=1, axis=0)
            ).sum(axis=0).sum(axis=-1))
            return ll_first + ll_rest
        else:
            return ll_first

    def dt(self, time_idx: int) -> float:
        """
        Return the k-th time increment, t_k - t_{k-1}.
        Raises an error if k == 0.
        :param time_idx: The index (k).
        :return:
        """
        if time_idx == 0 or time_idx >= self.num_times:
            raise IndexError("Can't get time increment at index {}.".format(time_idx))
        else:
            return self.times[time_idx] - self.times[time_idx - 1]

    def time_scaled_variance(self, time_idx: int, var_1: float, var: float) -> float:
        """
        Return the k-th time incremental variance.
        :param time_idx: the index to query (corresponding to k).
        :param var_1: The value of (tau_1)^2, the variance-scaling term of the first observed timepoint.
        :param var: The value of (tau)^2, the variance-scaling term of the underlying Gaussian process.
        :return: the kth variance term (t_k - t_(k-1)) * tau^2.
        """

        if time_idx == 0:
            return var_1
        elif time_idx < len(self.times):
            return var * self.dt(time_idx)
        else:
            raise IndexError("Can't reference time at index {}.".format(time_idx))
