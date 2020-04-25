import math
from util.io.logger import logger
from algs.base import AbstractModelSolver, compute_read_likelihoods
from abc import ABCMeta, abstractmethod


# ================================================================================================
# =============== Variational Inference (for learning approximate posteriors) ====================
# ================================================================================================

class AbstractGradientVISolver(AbstractModelSolver, metaclass=ABCMeta):
    def __init__(self, generative_model, data, variational_posterior):
        super().__init__(generative_model, data)
        self.frag_errors = compute_read_likelihoods(generative_model, data)
        self.posterior = variational_posterior

    def solve(self, iters=100, thresh=1e-5):
        """
        Runs the VI algorithm using the specified ELBO + optimization implementation.
        At the end, the solver's posterior is the optimized solution.
        :param iters: The number of iterations.
        """
        prev_obj = float("-inf")
        k = 0

        # Iterate until convergence in ELBO or specified number of iterations.
        for i in range(iters):
            # One step of VI.
            self.variational_iter()
            obj = self.posterior.elbo(self.model)

            # Terminate early if converged.
            if (not math.isinf(prev_obj)) and (abs(prev_obj - obj) < thresh):
                logger.debug("Convergence criterion met; terminating optimization early.".format(t=thresh))
                break
            prev_obj = obj

            # Debugging checkpoint.
            if k % 100 == 0:
                logger.debug("(Iteration {})  ELBO value: {}".format(k, obj))
            k += 1

        logger.debug("Finished {} iterations.".format(k))

    @abstractmethod
    def variational_iter(self):
        pass


class AbstractVariationalPosterior(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, num_samples=1):
        """
        Returns a sample from this posterior distribution.
        :return: the specified number of (abundances, fragment frequencies) tuples.
        """
        pass

    @abstractmethod
    def elbo(self, model):
        pass


# ========================== Implementations ===========================
class SecondOrderVariationalGradientSolver(AbstractGradientVISolver):
    """
      The VI formulation based on the second-order Taylor approximation (Hessian calculation).
    """
    def __init__(self, generative_model, data, posterior):
        super().__init__(generative_model, data, posterior)

    def variational_iter(self):
        # ==== update gaussians. ("prev_mean" means "previous in time", not "previous in iterations").
        prev_mean = self.model.mu
        for t_idx in range(len(self.model.times())):
            frag_probs = np.sum(self.posterior.frag_freqs[t_idx], axis=1)  # sum the frequencies over reads: (sum_i \phi^{t,i}_{f})
            (H, V) = self.gradient_update(frag_probs, prev_mean)
            self.posterior.covariances[t_idx] = np.linalg.inv(
                (1 / math.pow(self.model.time_scale(t_idx), 2)) * np.eye(self.model.num_strains()) - H)
            self.posterior.means[t_idx] = prev_mean + np.matmul(self.posterior.covariances[t_idx], V)

        # ==== update fragment probabilities.
        for t_idx in range(len(self.model.times())):
            E_log_z = log_frequency_expectations(self.model, self.posterior.means, t_idx)
            for r_idx in range(len(self.data)):
                self.posterior.frag_freqs[t_idx][r_idx, :] = self.frag_errors[t_idx][r_idx] * np.exp(E_log_z)

    def gradient_update(self, frag_probs, center):
        H = np.zeros(self.model.num_strains(), self.model.num_strains)
        V = np.zeros(self.model.num_strains(), 1)
        for frag in range(len(self.model.fragment_space)):
            H_f = frag_probs[frag] * map_hessian(center, self.model.W, frag)
            V_f = frag_probs[frag] * np.transpose(map_gradient(center, self.model.W, frag))
            H = H + H_f
            V = V + V_f
        return (H, V)


class SecondOrderVariationalPosterior(AbstractVariationalPosterior):
    """
      The variational posterior of conditional distributions on the Gaussian trajectory based on
      the second-order Taylor approximation (Hessian calculation).
      TODO: implement sampling and ELBO.
    """
    def __init__(self, means, covariances, frag_freqs):
        self.means = means
        self.covariances = covariances
        self.frag_freqs = frag_freqs

    def sample(self, num_samples=1):
        raise NotImplementedError("TODO implement me!")

    def elbo(self, model):
        raise NotImplementedError("TODO implement me!")


# ===================================================================
# ========================= Helper functions ========================
# ===================================================================

def log_frequency_expectations(model, means, t_idx):
    # Mathematically non-rigorous. Only should work if covariances are very tiny, e.g. O(1/sqrt(N)).
    return np.log(model.W * softmax(means[t_idx]))


def gradient_update(model, frag_probs, center):
    H = np.zeros(model.num_strains(), model.num_strains)
    V = np.zeros(model.num_strains(), 1)
    for frag in range(len(model.fragment_space)):
        H_f = frag_probs[frag] * map_hessian(center, model.W, frag)
        V_f = frag_probs[frag] * np.transpose(map_gradient(center, model.W, frag))
        H = H + H_f
        V = V + V_f
    return (H, V)


def map_gradient(center, W, f):
    # Outputs a row vector.
    deriv = np.matmul(W[f, :]) * softmax_derivative(x=center)
    return deriv / np.matmul(W[f, :], softmax(x=center))


def map_hessian(center, W, f):
    N = len(center)
    dot_product = np.matmul(W[f, :], softmax(x=center))
    d = map_gradient(center, W, f)
    tensor = softmax_second_derivative_tensor(center)
    second_deriv = np.zeros((N, N))
    for k in range(N):
        second_deriv = second_deriv + (W[f, k] * tensor[k, :, :])
    return (second_deriv / dot_product) - np.matmul(np.transpose(d), d)


def softmax_derivative(x):
    s = softmax(x)
    N = len(s)
    deriv = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            deriv[i][j] = s[i] * (delta(i, j) - s[j])
    return deriv
    # sbar = 1 - s
    # return s * np.transpose(sbar) - (np.ones((N, N)) - np.eye(N)) * s[:, None]


def softmax_second_derivative_tensor(x):
    s = softmax(x)
    N = len(s)
    deriv = np.zeros((N, N, N))
    for k in range(N):
        for i in range(N):
            for j in range(N):
                # second derivative of sigma_k (with respect to x_i, x_j)
                deriv[s][i][j] = s[k] * (
                        ((delta(j, k) - s[j]) * (delta(i, k) - s[i]))
                        -
                        (s[i] * (delta(i, j) - s[j]))
                )
    return deriv


def delta(i, j):
    if i == j:
        return 1
    return 0


def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)