import numpy as np
from util.logger import logger
from algs.base import AbstractModelSolver, compute_frag_errors

# ===========================================================================================
# =============== Expectation-Maximization (for getting a MAP estimator) ====================
# ===========================================================================================

class EMSolver(AbstractModelSolver):
    """
    MAP estimation via Expectation-Maximization.
    """
    def __init__(self, generative_model, data, lr=0.001):
        super().__init__(generative_model, data)
        self.frag_errors = compute_frag_errors(self.model, self.data)
        self.lr = lr

    def solve(self, iters=100, thresh=1e-5, initialization=None):
        """
        Runs the EM algorithm on the instantiated data.
        :param iters: number of iterations.
        :param thresh: the threshold that determines the convergence criterion (implemented as Frobenius norm of abundances).
        :param initialization: A (T x S) matrix of time-series abundances. If not specified, set to all-zeros matrix.
        :return: The estimated abundances.
        """
        if initialization is None:
            # T x S matrix of time-indexed abundances.
            abundances = np.ones((len(self.model.times), len(self.model.bacteria_pop.strains)), dtype=float)
        else:
            abundances = initialization

        k = 0
        for i in range(iters):
            updated_abundances = self.em_update(abundances)

            diff = np.linalg.norm(updated_abundances - abundances, 'fro')

            has_converged = (diff < thresh)

            if has_converged:
                logger.debug("Convergence criterion ({t}) met; terminating optimization early.".format(t=thresh))
                break

            abundances = updated_abundances

            if k % 100 == 0:
                logger.debug("(Iteration {})  abundance difference: {}".format(k, diff))
            k += 1
        logger.debug("Finished {} iterations.".format(k))
        return abundances

    def em_update(self, abundances):
        updated_abundances = []

        for time_index, (guessed_abundances_at_t, reads_at_t) in \
                enumerate(zip(abundances, self.data)):

            ##############################
            # Compute the "Q" vector
            ##############################

            rel_abundances_motion_guess = self.model.generate_relative_abundances(abundances)
            time_indexed_fragment_frequencies_guess = self.model.generate_time_indexed_fragment_frequencies(
                rel_abundances_motion_guess[time_index])

            # Step 1
            v = []
            for read_index, read in enumerate(reads_at_t):
                read_v = np.multiply(self.frag_errors[time_index][read_index], time_indexed_fragment_frequencies_guess)
                read_v = read_v / sum(read_v)
                v.append(read_v)

            # Step 2
            v = np.asanyarray(v)
            v = sum(v)

            # Step 3
            q_guess = np.divide(v, time_indexed_fragment_frequencies_guess)

            ##############################
            # Compute the regularization term
            ##############################

            if time_index == 1:
                regularization_term = abundances[time_index] - abundances[time_index + 1]
            elif time_index == len(self.model.times) - 1:
                regularization_term = abundances[time_index] - abundances[time_index - 1]
            else:
                regularization_term = (2 * abundances[time_index] -
                                       abundances[time_index - 1] -
                                       abundances[time_index + 1])

            scaled_tau = self.model.time_scale(time_index) ** 2
            regularization_term *= -1 / scaled_tau

            ##############################
            # Compute the derivative of relative abundances at X^t
            # An S x S Jacobian
            ##############################

            sigma_prime = np.zeros((len(guessed_abundances_at_t), len(guessed_abundances_at_t)))
            for i in range(len(guessed_abundances_at_t)):
                for j in range(len(guessed_abundances_at_t)):
                    delta = 1 if i == j else 0
                    sigma_prime[i][j] = guessed_abundances_at_t[i] * (delta - guessed_abundances_at_t[j])

            ##############################
            # Compute the 'main' term
            ##############################

            W = self.model.fragment_frequencies

            main_term = np.matmul(np.matmul(np.transpose(sigma_prime), np.transpose(W)), q_guess)

            ##############################
            # Update our guess for the motion at this time step
            ##############################

            updated_abundances.append(
                abundances[time_index] + self.lr * (main_term + regularization_term)
            )

        return np.array(updated_abundances)
