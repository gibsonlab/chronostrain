from typing import Tuple

import jax
import jax.numpy as np
import jax.experimental.sparse as jsparse

from chronostrain.model.io import TimeSeriesReads
from chronostrain.model.generative import AbundanceGaussianPrior
from chronostrain.util.benchmarking import RuntimeEstimator
from chronostrain.inference.algs.base import AbstractModelSolver
from chronostrain.util.math import log_spspmm_exp

from chronostrain.logging import create_logger
from chronostrain.database import StrainDatabase

logger = create_logger(__name__)
spmm = jsparse.sparsify(np.matmul)


# ===========================================================================================
# =============== Expectation-Maximization (for computing a MAP estimator) ==================
# ===========================================================================================

def _sics_mode(dof: float, scale: float) -> float:
    return dof * scale / (dof + 2)


def sparse_ll_exp(x: jsparse.BCOO):
    return jsparse.BCOO(
        (np.exp(x.data), x.indices),
        shape=x.shape
    )


# noinspection PyPep8Naming
class EMSolver(AbstractModelSolver):
    """
    MAP estimation via Expectation-Maximization.
    """
    def __init__(self,
                 generative_model: AbundanceGaussianPrior,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 dtype='bfloat16',
                 lr: float = 1e-3):
        """
        Instantiates an EMSolver instance.

        :param generative_model: The underlying generative model with prior parameters.
        :param data: the observed read_frags, a time-indexed list of read collections.
        :param db: The StrainDatabase instance.
        :param lr: the learning rate (default: 1e-3)
        """
        super().__init__(generative_model, data, db)
        self.dtype = dtype
        self.lr = lr
        self.frag_freqs = sparse_ll_exp(self.model.fragment_frequencies_sparse)

        data_likelihoods = self.data_likelihoods
        self.data_p = [
            sparse_ll_exp(m)
            for m in data_likelihoods.matrices
        ]
        self.dts = np.diff(np.array(self.model.times))
        self.initialize_gradient_functions()

        # initialize likelihood matrix products
        self.strain_data_lls = [
            log_spspmm_exp(self.model.fragment_frequencies_sparse.T, data_ll_t)
            for data_ll_t in data_likelihoods.matrices
        ]

    def solve(self,
              iters: int = 1000,
              thresh: float = 1e-5,
              gradient_clip: float = 1e2,
              initialization=None,
              print_debug_every=200
              ):
        """
        Runs the EM algorithm on the instantiated read_frags.
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
            x = np.zeros(
                shape=[len(self.model.times), len(self.model.bacteria_pop.strains)]
            )
        else:
            x = initialization

        var_1 = _sics_mode(dof=self.model.tau_1_dof, scale=self.model.tau_1_scale)
        var = _sics_mode(dof=self.model.tau_dof, scale=self.model.tau_scale)

        logger.debug("EM algorithm started. (Gradient method, Target iterations={}, Threshold={})".format(
            iters,
            thresh
        ))
        time_est = RuntimeEstimator(total_iters=iters, horizon=print_debug_every)
        k = 0
        while k < iters:
            k += 1
            time_est.stopwatch_click()
            x_new, updated_var_1, updated_var = self.em_update(
                x,
                time_scaled_variance=np.concatenate([
                    np.expand_dims(var_1, axis=0),
                    var * self.dts
                ]),
                gradient_clip=gradient_clip
            )

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            diff = np.sqrt(np.square(x_new - x).sum())
            x = x_new
            var_1 = updated_var_1
            var = updated_var

            if k % print_debug_every == 0:
                from chronostrain.inference.algs.vi.base.util import log_mm_exp
                # Evaluate read_frags likelihood
                ll = self.model.log_likelihood_x(x)
                for t in range(self.model.num_times()):
                    log_y_t = jax.nn.log_softmax(x[t], axis=-1)
                    prod = log_mm_exp(
                        np.expand_dims(log_y_t, axis=0),
                        self.strain_data_lls[t]
                    )
                    ll += np.where(np.isinf(prod), 0.0, prod).sum()
                logger.info("Iteration {i} | time left: {t:.1f} min. | Learned Abundance Diff: {diff} | LL = {ll}".format(
                    i=k,
                    t=time_est.time_left() / 60000,
                    diff=diff,
                    ll=ll
                ))

            has_converged = (diff < thresh)
            if has_converged:
                logger.info("Convergence criterion ({th}) met; terminating optimization early.".format(th=thresh))
                break
        logger.info("Finished {k} iterations.".format(k=k))
        return x

    def initialize_gradient_functions(self):
        mu_0 = np.zeros(shape=(self.model.num_strains()), dtype=self.dtype)

        # @jax.jit
        def gaussian_gradient(x: np.ndarray, time_scaled_variances: np.ndarray):
            # x has shape (T,S)
            var_scaling = -1 / time_scaled_variances  # (T,)

            fwd_diffs = np.expand_dims(var_scaling, axis=1) * np.diff(
                np.concatenate([
                    np.expand_dims(mu_0, axis=0),  # (1xS)
                    x
                ]),
                axis=0
            )  # x_t - x_t-1, including x_1 - mu  --> (T, S)
            rev_diffs = np.expand_dims(var_scaling[1:], axis=1) * np.diff(-x, axis=0)  # x_{t-1} - x_t, starting with x_2 - x_1 --> (T-1, S)
            return fwd_diffs + np.concatenate([
                rev_diffs,
                np.zeros(shape=(1, x.shape[1]), dtype=self.dtype)
            ])
        self.gaussian_gradient = gaussian_gradient

        # @jax.jit
        def data_gradient(x_t: np.ndarray, frag_freq: jsparse.BCOO, data_p_t: jsparse.BCOO):
            """
            frag_freq is (F x S)
            data_p_t is (F x R)
            """
            y_t = jax.nn.softmax(x_t, axis=-1)  # (S)
            Z_t = frag_freq @ y_t  # (F,S) @ (S) --> (F), frag freq at time t
            print(Z_t)  # DEBUG

            Q = data_p_t * np.expand_dims(Z_t, axis=1)  # scale each row by \tilde{Z} --> (F,R)
            Q = Q / jsparse.bcoo_reduce_sum(Q, axes=[0]).todense()  # normalize each column --> (F,R)  TODO: check if this has nans. (it probably does)
            Q = jsparse.bcoo_reduce_sum(Q, axes=[1]) / Z_t  # take column sum and divide by 1 / Z --> (F,)

            print("****************")
            print(data_p_t)
            print(Q)
            print(Q.todense())
            print(Q.data)
            print("****************")

            sigmoid = y_t
            sigmoid_jacobian = np.diag(sigmoid) - np.outer(sigmoid, sigmoid)  # (S,S)

            # ========== DEBUG
            # _A = sigmoid_jacobian @ frag_freq.T
            # _B = Q.todense()
            # print(_A[0])
            # print(_B)
            # print(np.dot(_A[0], _B))
            # np.savez("TEST/arrays.npz", A=_A, B=_B)
            #
            # print(sigmoid_jacobian @ frag_freq.T @ Q)
            return sigmoid_jacobian @ frag_freq.T @ Q
        self.data_gradient = data_gradient

    def em_update(
            self,
            x: np.ndarray,
            time_scaled_variance: np.ndarray,
            gradient_clip: float
    ) -> Tuple[np.ndarray, float, float]:
        x_gradient = (
            self.gaussian_gradient(x, time_scaled_variance)
            +
            np.stack([
                self.data_gradient(
                    x[t],
                    self.frag_freqs,
                    self.data_p[t],
                    debug=(t == 3)
                )
                for t in range(x.shape[0])
            ], axis=0)
        )

        # ==== Gradient clipping.
        x_gradient = np.where(
            np.abs(x_gradient) > gradient_clip,
            np.sign(x_gradient) * gradient_clip,
            x_gradient
        )

        new_x = x + self.lr * x_gradient

        # ==== Estimate variances from new posterior.
        updated_var_1, updated_var = self.estimate_posterior_variances(new_x)
        return new_x, updated_var_1, updated_var

    def estimate_posterior_variances(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Outputs the posterior modes (maximum posterior likelihood), using the conjugacy of SICS/Gaussian distributions.

        :param x: a (T x S) tensor of gaussians, representing a realization of the S-axisensional brownian motion.
        """
        diffs_1 = x[0, :]
        dof_1 = self.model.tau_1_dof + diffs_1.size
        scale_1 = (1 / dof_1) * (
            self.model.tau_1_dof * self.model.tau_1_scale
            + np.sum(np.square(diffs_1))
        ).item()

        diffs = (x[1:, :] - x[:-1, :]) * np.expand_dims(np.power(np.array(
            [self.model.dt(t_idx) for t_idx in range(1, self.model.num_times())],
        ), -0.5), axis=1)
        dof = self.model.tau_dof + diffs.size
        scale = (1 / dof) * (
            self.model.tau_dof * self.model.tau_scale
            + np.sum(np.square(diffs))
        ).item()

        return _sics_mode(dof_1, scale_1), _sics_mode(dof, scale)
