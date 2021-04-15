import torch
from typing import List

from chronostrain.config import cfg
from chronostrain.util.data_cache import CacheTag
from chronostrain.algs.em import EMSolver
from chronostrain.util.benchmarking import RuntimeEstimator
from chronostrain.algs.base import AbstractModelSolver
from chronostrain.model.io import TimeSeriesReads
from chronostrain.model.generative import GenerativeModel
from . import logger

from torch.nn.functional import softmax


# ===========================================================================================
# =============== Expectation-Maximization (for computing a MAP estimator) ==================
# ===========================================================================================


class EMAlternateSolver(AbstractModelSolver):
    """
    Alternating maximization scheme for MAP estimator (X, Strains) given Reads.
    """
    def __init__(
            self,
            generative_model: GenerativeModel,
            data: TimeSeriesReads,
            cache_tag: CacheTag,
            lr: float = 1e-3):
        """
        Instantiates an EMAlternateSolver instance.

        :param generative_model: The underlying generative model with prior parameters.
        :param data: the observed data, a time-indexed list of read collections.
        :param lr: the learning rate (default: 1e-3)
        """
        super().__init__(generative_model, data, cache_tag)
        self.lr = lr
        self.T = len(self.data)
        self.num_reads_per_t = torch.tensor([len(self.data[t]) for t in range(self.T)], device=cfg.torch_cfg.device)

    def solve(self,
              max_iters: int = 1000,
              x_opt_thresh: float = 1e-5,
              print_debug_every=200
              ):
        em_solver = EMSolver(generative_model=self.model,
                             data=self.data,
                             cache_tag=self.cache_tag,
                             lr=self.lr)

        logger.debug("Alternating optimization algorithm started. (Max iterations={})".format(
            max_iters
        ))

        logger.debug("Initializing with EM instance...")
        x = em_solver.solve(iters=max_iters, thresh=x_opt_thresh, print_debug_every=1000)
        strains = self.solve_strain_asgns(x)

        iter_idx = 1
        time_est = RuntimeEstimator(total_iters=max_iters, horizon=print_debug_every)
        while iter_idx <= max_iters:
            time_est.stopwatch_click()
            x = self.solve_abundances(x_init=x,
                                      strain_assignments=strains,
                                      iters=100,  # Hard-coded.
                                      thresh=x_opt_thresh,
                                      print_debug_every=1000)
            strains_next = self.solve_strain_asgns(x)

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            num_reassigned_strains = torch.tensor(
                [torch.sum(strains[t] != strains_next[t]) for t in range(self.T)],
                dtype=torch.bool
            ).sum()

            if num_reassigned_strains == 0:
                logger.info("Convergence criterion met after {} iterations.".format(iter_idx))
                break
            else:
                print("\t# strains reassigned: {}".format(num_reassigned_strains))

            if iter_idx % print_debug_every == 0:
                logger.debug("Iteration {i} | time left: {t:.1f} min.".format(
                    i=iter_idx,
                    t=time_est.time_left() / 60000
                ))

            strains = strains_next
            iter_idx += 1
        return x, strains

    def solve_abundances(self,
                         x_init: torch.Tensor,
                         strain_assignments: List[torch.Tensor],
                         iters: int = 1000,
                         thresh: float = 1e-5,
                         print_debug_every=200
                         ) -> torch.Tensor:
        """
        :param strain_assignments: The 2-d array containing the strain assigned to each read
        (therefore, size must match input data).
        :param iters: number of iterations.
        :param thresh: the threshold that determines the convergence criterion (implemented as Frobenius norm of
        abundances).
        :param x_init: A (T x S) matrix of time-series abundances. If not specified, set to all-zeros matrix.
        :param print_debug_every: The number of iterations to skip between debug logging summary.
        :return: The estimated abundances
        """
        brownian_motion = x_init

        logger.debug("Optimizing latent Gaussians. (Gradient method, Target iterations={}, Threshold={})".format(
            iters,
            thresh
        ))
        time_est = RuntimeEstimator(total_iters=iters, horizon=print_debug_every)
        k = 1
        while k <= iters:
            time_est.stopwatch_click()

            """
            Strain assignment indicators 
            ([t][s] entry equals the number of reads at time t assigned to strain s)
            """
            strain_asgn_indicators = torch.zeros(size=brownian_motion.size(), device=cfg.torch_cfg.device)
            for t in range(self.T):
                strain_t = torch.tensor(strain_assignments[t], device=cfg.torch_cfg.device)
                uniques, counts = torch.unique(strain_t, return_counts=True)
                for s, count in zip(uniques, counts):
                    strain_asgn_indicators[t][s] = count

            """
            Update step
            """
            updated_brownian_motion = self.bm_gradient_update(
                strain_asgn_indicators,
                brownian_motion
            )
            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            """
            Calculate difference and check termination condition (diff < thresh).
            """
            diff = torch.norm(
                softmax(updated_brownian_motion, dim=1) - softmax(brownian_motion, dim=1),
                p='fro'
            )

            has_converged = (diff < thresh)
            if has_converged:
                logger.debug("Convergence criterion ({th}) met; terminating optimization early.".format(th=thresh))
                break
            brownian_motion = updated_brownian_motion

            """
            Status message.
            """
            if k % print_debug_every == 0:
                logger.debug("Iteration {i} | time left: {t:.1f} min. | Learned Abundance Diff: {diff}".format(
                    i=k,
                    t=time_est.time_left() / 60000,
                    diff=diff
                ))
            k += 1
        logger.info("Finished {k} iterations.".format(k=k))
        abundances = softmax(brownian_motion, dim=1)
        return abundances

    def bm_gradient_update(
            self,
            strain_asgn_indicators: torch.Tensor,
            x: torch.Tensor
    ):
        x_gradient = torch.zeros(size=x.size(), device=cfg.torch_cfg.device)  # T x S tensor.
        # ====== Gaussian part
        if self.T > 1:
            for t in range(self.T):
                if t == 0:
                    variance_scaling = -1 / (self.model.tau_1 ** 2)
                    x_gradient[t] = variance_scaling * ((2 * x[0]) - x[1] - self.model.mu)
                elif t == self.T-1:
                    variance_scaling = -1 / ((self.model.tau * (self.model.times[t] - self.model.times[t-1])) ** 2)
                    x_gradient[t] = variance_scaling * (x[self.T-1] - x[self.T-2])
                else:
                    variance_scaling_prev = -1 / ((self.model.tau * (self.model.times[t] - self.model.times[t-1])) ** 2)
                    variance_scaling_next = -1 / ((self.model.tau * (self.model.times[t+1] - self.model.times[t])) ** 2)
                    x_gradient[t] = variance_scaling_prev * (x[t] - x[t-1]) + variance_scaling_next * (x[t] - x[t+1])

        # ====== Sigmoidal part
        y = softmax(x, dim=1)
        x_gradient = x_gradient + strain_asgn_indicators - (y * self.num_reads_per_t[:, None])

        # ==== Gradient update.
        updated_x = x + (self.lr * x_gradient)

        # # ==== Re-center to zero to prevent drift. (Adding constant to each component does not change softmax.)
        # updated_x = updated_x - (updated_x.mean() * torch.ones(size=updated_x.size(), device=cfg.torch_cfg.device))

        return updated_x

    def solve_strain_asgns(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        :param x: A (T x S) tensor of latent representation of abundances.
        :return: A 2-d array of strain assignments for each read (matches read/data shape (T x N_t)).
        """
        assignments = []
        y = softmax(x, dim=1)
        for t in range(self.T):
            y_t = y[t]
            # GOAL: End up with an (N x S) tensor, scale each (s-th) column by y_t[s] then do argmax(dim=1).
            # second term: (N x F) * (F x S) -> (N x S)
            prob = y_t[:, None].t() * self.read_likelihoods[t].t().mm(self.model.get_fragment_frequencies())
            argmax_asgn_t = torch.argmax(prob, dim=1)
            assignments.append(argmax_asgn_t)
        return assignments
