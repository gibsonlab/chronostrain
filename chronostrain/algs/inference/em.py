from typing import Tuple

import jax
import jax.numpy as np
import jax.experimental.sparse as jsparse

from chronostrain.config import cfg
from chronostrain.model.io.reads import TimeSeriesReads
from chronostrain.model.generative import GenerativeModel
from chronostrain.util.benchmarking import RuntimeEstimator
from .base import AbstractModelSolver

from chronostrain.logging import create_logger
from chronostrain.database import StrainDatabase
from chronostrain.util.math import *

logger = create_logger(__name__)
spmm = jsparse.sparsify(np.matmul)


# ===========================================================================================
# =============== Expectation-Maximization (for computing a MAP estimator) ==================
# ===========================================================================================

def _sics_mode(dof: float, scale: float) -> float:
    return dof * scale / (dof + 2)


# noinspection PyPep8Naming
class EMSolver(AbstractModelSolver):
    """
    MAP estimation via Expectation-Maximization.
    """
    def __init__(self,
                 generative_model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 lr: float = 1e-3):
        """
        Instantiates an EMSolver instance.

        :param generative_model: The underlying generative model with prior parameters.
        :param data: the observed data, a time-indexed list of read collections.
        :param db: The StrainDatabase instance.
        :param lr: the learning rate (default: 1e-3)
        """
        super().__init__(generative_model, data, db)
        self.lr = lr

    def solve(self,
              iters: int = 1000,
              thresh: float = 1e-5,
              gradient_clip: float = 1e2,
              initialization=None,
              print_debug_every=200
              ):
        """
        Runs the EM algorithm on the instantiated data.
        :param iters: number of iterations.
        :param thresh: the threshold that determines the convergence criterion (implemented as Frobenius norm of
        abundances).
        :param gradient_clip: An upper bound on the Frobenius norm of the underlying GP trajectory
        (as a T x S matrix).
        :param initialization: A (T x S) matrix of time-series abundances. If not specified, set to all-zeros matrix.
        :param print_debug_every: The number of iterations to skip between debug logging summary.
        :return: The estimated abundances
        """

        if initialization is None:
            # T x S array representing a time-indexed, S-axisensional brownian motion.
            brownian_motion = np.ones(
                shape=[len(self.model.times), len(self.model.bacteria_pop.strains)]
            )
        else:
            brownian_motion = initialization

        var_1 = _sics_mode(dof=self.model.tau_1_dof, scale=self.model.tau_1_scale)
        var = _sics_mode(dof=self.model.tau_dof, scale=self.model.tau_scale)

        logger.debug("EM algorithm started. (Gradient method, Target iterations={}, Threshold={})".format(
            iters,
            thresh
        ))
        time_est = RuntimeEstimator(total_iters=iters, horizon=print_debug_every)
        k = 0
        softmax_diff = float("inf")
        while k < iters:
            k += 1
            time_est.stopwatch_click()
            updated_brownian_motion, updated_var_1, updated_var = self.em_update(
                brownian_motion,
                var_1=var_1,
                var=var,
                gradient_clip=gradient_clip
            )

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            softmax_diff = np.linalg.norm(
                jax.nn.softmax(updated_brownian_motion, axis=1) - jax.nn.softmax(brownian_motion, axis=1),
                p='fro'
            ).item()

            has_converged = (softmax_diff < thresh)
            if has_converged:
                logger.info("Convergence criterion ({th}) met; terminating optimization early.".format(th=thresh))
                break
            brownian_motion = updated_brownian_motion
            var_1 = updated_var_1
            var = updated_var

            if k % print_debug_every == 0:
                logger.info("Iteration {i} | time left: {t:.1f} min. | Learned Abundance Diff: {diff}".format(
                    i=k,
                    t=time_est.time_left() / 60000,
                    diff=softmax_diff
                ))
        logger.info("Finished {k} iterations. | Abundance diff = {diff} | var_1 = {var_1} | var = {var}".format(
            k=k,
            diff=softmax_diff,
            var_1=var_1,
            var=var
        ))

        return jax.nn.softmax(brownian_motion, axis=1), var_1, var

    def em_update(
            self,
            x: np.ndarray,
            var: float,
            var_1: float,
            gradient_clip: float
    ) -> Tuple[np.ndarray, float, float]:
        T, S = x.shape
        x_gradient = np.zeros(size=x.shape)  # T x S tensor.

        # ====== Gaussian part
        if T > 1:
            for t in range(T):
                if t == 0:
                    variance_scaling = -1 / self.model.time_scaled_variance(0, var_1=var_1, var=var)
                    x_gradient[t] = variance_scaling * ((2 * x[0]) - x[1] - self.model.mu)
                elif t == T-1:
                    variance_scaling = -1 / self.model.time_scaled_variance(t, var_1=var_1, var=var)
                    x_gradient[t] = variance_scaling * (x[T-1] - x[T-2])
                else:
                    variance_scaling_prev = -1 / self.model.time_scaled_variance(t, var_1=var_1, var=var)
                    variance_scaling_next = -1 / self.model.time_scaled_variance(t+1, var_1=var_1, var=var)
                    x_gradient[t] = variance_scaling_prev * (x[t] - x[t-1]) + variance_scaling_next * (x[t] - x[t+1])

        # ====== Sigmoidal part
        y = jax.nn.softmax(x, axis=1)
        frag_log_freq = self.model.fragment_frequencies_sparse
        frag_freq = jsparse.BCOO(
            (np.exp(frag_log_freq.data), frag_log_freq.indices),
            shape=frag_log_freq.shape
        )
        for t in range(T):
            # Scale each row by Z_t, and normalize.
            if cfg.model_cfg.use_sparse:
                Z_t = np.exp(densesp_mm(y[t], frag_freq))
                Q = column_normed_row_sum(
                    scale_row(self.data_likelihoods.matrices[t], Z_t)
                ) / Z_t

                sigmoid = y[t]
                sigmoid_jacobian = np.diag(sigmoid) - np.outer(sigmoid, sigmoid)  # symmetric matrix.

                x_gradient[t] = x_gradient[t] + sigmoid_jacobian.mv(
                    jsparse.bcoo_multiply_dense(
                        self.model.fragment_frequencies_sparse.T,
                        Q
                    )
                )
            else:
                raise NotImplementedError("Dense version of this is not implemented for JAX.")

        # ==== Gradient clipping.
        x_gradient[x_gradient > gradient_clip] = gradient_clip
        x_gradient[x_gradient < -gradient_clip] = -gradient_clip

        updated_x: np.ndarray = x + self.lr * x_gradient

        # ==== Estimate variances from new posterior.
        updated_var_1, updated_var = self.estimate_posterior_variances(updated_x)
        return updated_x, updated_var_1, updated_var

    def estimate_posterior_variances(self, x) -> Tuple[float, float]:
        """
        Outputs the posterior modes (maximum posterior likelihood), using the conjugacy of SICS/Gaussian distributions.

        :param x: a (T x S) tensor of gaussians, representing a realization of the S-axisensional brownian motion.
        """
        diffs_1 = x[0, :] - self.model.mu
        dof_1 = self.model.tau_1_dof + diffs_1.numel()
        scale_1 = (1 / dof_1) * (
            self.model.tau_1_dof * self.model.tau_1_scale
            + np.sum(np.square(diffs_1))
        )

        diffs = (x[1:, :] - x[:-1, :]) * np.expand_dims(np.power(np.array(
            [self.model.dt(t_idx) for t_idx in range(1, self.model.num_times())],
        ), -0.5), axis=1)
        dof = self.model.tau_dof + diffs.numel()
        scale = (1 / dof) * (
            self.model.tau_dof * self.model.tau_scale
            + np.sum(np.square(diffs))
        )

        return _sics_mode(dof_1, scale_1), _sics_mode(dof, scale)
