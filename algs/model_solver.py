import math
import numpy as np

# ===============
# An implementation of the proposed algorithms.
# ===============

class GenerativeModel:
    def __init__(self, times, mu, tau_1, tau, W, strains, fragments):
        self.times = times
        self.mu = mu
        self.tau_1 = tau_1
        self.tau = tau
        self.W = W
        self.strains = strains
        self.fragments = fragments

    def num_strains(self):
        return len(self.strains)

    def num_fragments(self):
        # Should output F, the total number of possible fragments in the database.
        raise NotImplementedError()

    def time_scale(self, time_idx):
        if time_idx == 0:
            return self.tau_1
        if time_idx < len(self.times):
            return self.tau * (self.times[time_idx] - self.times[time_idx] - 1)
        else:
            return IndexError("Can't reference time at index {}.".format(time_idx))

    def num_times(self):
        return len(self.times)



def variational_learn(model, reads, tol=1e-5):
    # Initialization
    frag_errors = compute_frag_errors(model, reads)
    means = [np.zeros(model.strains, 1) for t in model.times]
    covariances = [model.time_scale(k) * np.eye(model.strains) for k in range(model.num_times())]
    frag_freqs = [
        (1 / model.num_fragments()) * np.ones(model.num_fragments, len(model.reads[t]))
        for t in model.times
    ]

    # Update
    prev_loss = None
    loss_change = None
    while (loss_change is None or loss_change < tol):
        loss = variational_update(model, reads, means, covariances, frag_freqs, frag_errors)
        loss_change = np.abs(prev_loss - loss)
        prev_loss = loss

    return means, covariances, frag_freqs


def compute_frag_errors(model, reads):
    raise NotImplementedError()


def compute_model_ELBO(model, reads, means, covariances, frag_freqs):
    raise NotImplementedError()


def variational_update(model, reads, means, covariances, frag_freqs, frag_errors):
    # ==== update gaussians.
    prev_mean = model.mu
    for t_idx in range(len(model.times())):
        frag_probs = np.sum(frag_freqs[t_idx], axis=1)  # sum the frequencies over reads (sum_i \phi^{t,i}_{f})
        (H, V) = gradient_update(model, frag_probs, prev_mean)
        covariances[t_idx] = np.linalg.inv((1 / math.pow(model.time_scale(t_idx), 2)) * np.eye(model.num_strains()) - H)
        means[t_idx] = prev_mean + np.matmul(covariances[t_idx], V)

    # ==== update fragment probabilities.
    for t_idx in range(len(model.times())):
        E_log_z = log_frequency_expectations(model, means, covariances, t_idx)
        for r_idx in range(len(model.reads())):
            frag_freqs[t_idx][r_idx, :] = frag_errors[t_idx][r_idx] * np.exp(E_log_z)

    # ==== Return ELBO loss.
    return compute_model_ELBO(model, reads, means, covariances, frag_freqs)


def log_frequency_expectations(model, means, covariances, t_idx):
    # Mathematically non-rigorous. Only should work if covariances are very tiny, e.g. O(1/sqrt(N)).
    return np.log(model.W * softmax(means[t_idx]))


def gradient_update(model, frag_probs, center):
    H = np.zeros(model.num_strains(), model.num_strains)
    V = np.zeros(model.num_strains(), 1)
    for frag in range(len(model.fragments)):
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
            deriv[i][j] = s[i] * (delta(i,j) - s[j])
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
                        ((delta(j,k) - s[j]) * (delta(i,k) - s[i]))
                        -
                        (s[i] * (delta(i,j) - s[j]))
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
