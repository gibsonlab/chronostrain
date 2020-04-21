from model.generative import softmax
from util.benchmarking import RuntimeEstimator
from util.logger import logger
from algs.base import AbstractModelSolver, compute_frag_errors

from typing import List
from model.reads import SequenceRead
from model.generative import GenerativeModel

import torch

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

# ===========================================================================================
# =============== Expectation-Maximization (for getting a MAP estimator) ====================
# ===========================================================================================


class EMSolver(AbstractModelSolver):
    """
    MAP estimation via Expectation-Maximization.
    """
    def __init__(self, generative_model: GenerativeModel,
                 data: List[List[SequenceRead]],
                 lr: int = 0.001,
                 device=default_device):

        super().__init__(generative_model, data)
        self.frag_errors = compute_frag_errors(self.model, self.data)
        self.lr = lr
        self.device = device

    def solve(self, iters=100, thresh=1e-5, initialization=None, print_debug_every=10):
        """
        Runs the EM algorithm on the instantiated data.
        :param iters: number of iterations.
        :param thresh: the threshold that determines the convergence criterion (implemented as Frobenius norm of abundances).
        :param initialization: A (T x S) matrix of time-series abundances. If not specified, set to all-zeros matrix.
        :return: The estimated abundances
        """

        if initialization is None:
            # T x S array of time-indexed abundances.
            brownian_motion = torch.ones(len(self.model.times), len(self.model.bacteria_pop.strains), device=self.device)
        else:
            brownian_motion = initialization

        time_est = RuntimeEstimator(total_iters=iters, horizon=100)
        k = 1
        while k <= iters:
            time_est.stopwatch_click()
            updated_brownian_motion = self.em_update(brownian_motion)
            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            diff = torch.norm(updated_brownian_motion - brownian_motion, p='fro')

            has_converged = (diff < thresh)
            if has_converged:
                logger.info("Convergence criterion ({th}) met; terminating optimization early.".format(th=thresh))
                break
            brownian_motion = updated_brownian_motion

            if k % print_debug_every == 0:
                logger.info("Iteration {i} | time left: {t} min. | Brownian Motion Diff: {diff}".format(
                    i=k,
                    t=time_est.time_left() // 60,
                    diff=diff
                ))
            k += 1
        logger.info("Finished {k} iterations.".format(k=k-1))

        abundances = [softmax(gaussian) for gaussian in brownian_motion]
        normalized_abundances = torch.stack(abundances).to(self.device)
        return normalized_abundances

    def em_update(self, brownian_motion: torch.Tensor) -> torch.Tensor:

        rel_abundances_motion_guess = [softmax(gaussian) for gaussian in brownian_motion]

        updated_brownian_motion = torch.ones(len(brownian_motion), len(brownian_motion[0]))

        for time_index, (guessed_gaussian_at_t, reads_at_t) in \
                enumerate(zip(brownian_motion, self.data)):

            ##############################
            # Compute the "Q" vector
            ##############################

            time_indexed_fragment_frequencies_guess = self.model.strain_abundance_to_frag_abundance(
                rel_abundances_motion_guess[time_index])

            # Step 1
            v = []
            for read_index, read in enumerate(reads_at_t):
                read_v = torch.mul(self.frag_errors[time_index][read_index], time_indexed_fragment_frequencies_guess)
                read_v = read_v / sum(read_v)
                v.append(read_v)

            # Step 2
            v = torch.stack(v).to(self.device)
            v = sum(v)

            # Step 3
            # TODO: Sometimes we get a divide by zero error here (particularly when a strain abundance
            #  is thought to be zero)
            q_guess = torch.div(v, time_indexed_fragment_frequencies_guess)

            ##############################
            # Compute the regularization term
            ##############################

            if time_index == 1:
                regularization_term = brownian_motion[time_index] - brownian_motion[time_index + 1]
            elif time_index == len(self.model.times) - 1:
                regularization_term = brownian_motion[time_index] - brownian_motion[time_index - 1]
            else:
                regularization_term = (2 * brownian_motion[time_index] -
                                       brownian_motion[time_index - 1] -
                                       brownian_motion[time_index + 1])

            scaled_tau = self.model.time_scale(time_index) ** 2
            regularization_term *= -1 / scaled_tau

            ##############################
            # Compute the derivative of relative abundances at X^t
            # An S x S Jacobian
            ##############################

            sigma_prime = torch.zeros(len(guessed_gaussian_at_t), len(guessed_gaussian_at_t), device=self.device)
            for i in range(len(guessed_gaussian_at_t)):
                for j in range(len(guessed_gaussian_at_t)):
                    delta = 1 if i == j else 0
                    sigma_prime[i][j] = guessed_gaussian_at_t[i] * (delta - guessed_gaussian_at_t[j])

            ##############################
            # Compute the 'main' term
            ##############################

            W = self.model.get_fragment_frequencies()
            main_term = torch.matmul(torch.matmul(sigma_prime.t(), W.t()), q_guess)

            ##############################
            # Update our guess for the motion at this time step
            ##############################

            updated_brownian_motion[time_index] = brownian_motion[time_index] + self.lr * (main_term + regularization_term)

        return torch.tensor(updated_brownian_motion)
