from util.benchmarking import RuntimeEstimator
from util.io.logger import logger
from algs.base import AbstractModelSolver, compute_read_likelihoods

from typing import List
from model.reads import SequenceRead
from model.generative import GenerativeModel

import torch
from torch.nn.functional import softmax

torch.set_default_dtype(torch.float64)

# ===========================================================================================
# =============== Expectation-Maximization (for getting a MAP estimator) ====================
# ===========================================================================================


class EMSolver(AbstractModelSolver):
    """
    MAP estimation via Expectation-Maximization.
    """
    def __init__(
            self,
            generative_model: GenerativeModel,
            data: List[List[SequenceRead]],
            torch_device,
            lr: float = 1e-3):
        """
        Instantiates an EMSolver instance.

        :param generative_model: The underlying generative model with prior parameters.
        :param data: the observed data, a time-indexed list of read collections.
        :param torch_device: the torch device to operate on. (Recommended: CUDA if available.)
        :param lr: the learning rate (default: 1e-3
        """
        super().__init__(generative_model, data)
        self.read_likelihoods = compute_read_likelihoods(self.model, self.data, logarithm=False, device=torch_device)
        self.lr = lr
        self.device = torch_device
        self.model.get_fragment_frequencies()

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
        :param gradient_clip: An upper bound on the Frobenius norm of the underlying GP trajectory (as a T x S matrix).
        :param initialization: A (T x S) matrix of time-series abundances. If not specified, set to all-zeros matrix.
        :param print_debug_every: The number of iterations to skip between debug logging summary.
        :return: The estimated abundances
        """
        if initialization is None:
            # T x S array representing a time-indexed, S-dimensional brownian motion.
            brownian_motion = torch.ones(
                size=[len(self.model.times), len(self.model.bacteria_pop.strains)],
                device=self.device
            )
        else:
            brownian_motion = initialization

        logger.debug("EM algorithm started. (Gradient method, Target iterations={}, Threshold={})".format(
            iters,
            thresh
        ))
        time_est = RuntimeEstimator(total_iters=iters, horizon=5)
        k = 1
        while k <= iters:
            time_est.stopwatch_click()
            updated_brownian_motion = self.em_update_new(brownian_motion, gradient_clip=gradient_clip)
            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            diff = torch.norm(
                softmax(updated_brownian_motion, dim=1) - softmax(brownian_motion, dim=1),
                p='fro'
            )

            has_converged = (diff < thresh)
            if has_converged:
                logger.info("Convergence criterion ({th}) met; terminating optimization early.".format(th=thresh))
                break
            brownian_motion = updated_brownian_motion

            if k % print_debug_every == 0:
                logger.info("Iteration {i} | time left: {t:.1f} min. | Learned Abundance Diff: {diff}".format(
                    i=k,
                    t=time_est.time_left() / 60000,
                    diff=diff
                ))
            k += 1
        logger.info("Finished {k} iterations.".format(k=k-1))

        abundances = [softmax(gaussian, dim=0) for gaussian in brownian_motion]
        normalized_abundances = torch.stack(abundances).to(self.device)
        return normalized_abundances

    def em_update_new(
            self,
            x: torch.Tensor,
            gradient_clip: float
    ):
        T, S = x.size()
        F = self.model.num_fragments()
        x_gradient = torch.zeros(size=x.size(), device=self.device)  # T x S tensor.

        # ====== Gaussian part
        if T > 1:
            for t in range(T):
                if t == 0:
                    variance_scaling = -1 / (self.model.tau_1 ** 2)
                    x_gradient[t] = variance_scaling * ((2 * x[0]) - x[1] - torch.zeros(S, device=self.device))
                elif t == T-1:
                    variance_scaling = -1 / ((self.model.tau * (self.model.times[t] - self.model.times[t-1])) ** 2)
                    x_gradient[t] = variance_scaling * (x[T-1] - x[T-2])
                else:
                    variance_scaling_prev = -1 / ((self.model.tau * (self.model.times[t] - self.model.times[t-1])) ** 2)
                    variance_scaling_next = -1 / ((self.model.tau * (self.model.times[t+1] - self.model.times[t])) ** 2)
                    x_gradient[t] = variance_scaling_prev * (x[t] - x[t-1]) + variance_scaling_next * (x[t] - x[t+1])

        # ====== Sigmoidal part
        y = softmax(x, dim=1)
        for t in range(T):
            # Scale each row by Z_t, and normalize.
            Z_t = self.model.strain_abundance_to_frag_abundance(y[t].view(S, 1))
            Q = self.get_frag_likelihoods(t) * Z_t
            Q = (Q / Q.sum(dim=0)[None, :]).sum(dim=1) / Z_t.view(F)

            sigmoid = y[t]
            sigmoid_jacobian = torch.ger(sigmoid, 1-sigmoid) - \
                               torch.diag(sigmoid).mm(
                                   torch.ones(S, S, device=self.device) - torch.eye(S, device=self.device)
                               )

            x_gradient[t] = x_gradient[t] + sigmoid_jacobian.mv(
                self.model.get_fragment_frequencies().t().mv(Q)
            )

        # ==== Gradient clipping.
        grad_t_norm = x_gradient.norm(p=2).item()
        if grad_t_norm > gradient_clip:
            x_gradient = x_gradient * gradient_clip / grad_t_norm

        # ==== Re-center to zero to prevent drift.
        updated_x = x + self.lr * x_gradient
        updated_x = updated_x - (updated_x.mean() * torch.ones(size=updated_x.size(), device=self.device))

        return updated_x

    def get_frag_likelihoods(self, t: int):
        """
        Look up the fragment error matrix for timeslice t.

        :param t: the time index (not the actual value).
        :return: An (F x N) matrix representing the read likelihoods according to the error model.
        """
        return self.read_likelihoods[t]

    def read_error_projections(self, t: int, frag_abundance: torch.Tensor) -> torch.Tensor:
        """
        :param t: the time index.
        :param frag_abundance: The vector of fragment abundances Z_t.
        :return: The vector of linear projections [<E_1^t, Z_t> , ..., <E_N^t, Z_t>].
        """
        # (N x F) matrix, applied to an F-dimensional vector.
        return self.read_likelihoods[t].t().mv(frag_abundance)
