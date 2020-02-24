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



def variational_learn(model, reads, tol=1e-6):

    # Initialization
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
        loss = variational_update(model, reads, means, covariances, frag_freqs)

    raise NotImplementedError()

def variational_update(model, reads, means, covariances, frag_freqs):
    # update gaussians.
    prev_mean = model.mu
    for k, t in enumerate(model.times):
        frag_probs = np.sum(frag_freqs[k], axis=1)  # TODO double check that sum is over correct axis.
        (H, V) = gradient_update(model, frag_probs, prev_mean)
        covariances[k] = np.linalg.inv((1 / math.pow(model.time_scale(k), 2)) * np.eye(model.num_strains()) - H)
        means[k] = prev_mean + np.matmul(covariances[k], V)

    # update fragment probabilities.
    ##== TODO

    # Return ELBO loss.
    raise NotImplementedError()


def gradient_update(model, frag_probs, center):
    H = np.zeros(model.num_strains(), model.num_strains)
    V = np.zeros(model.num_strains(), 1)
    for frag in range(len(model.fragments)):
        H_f = frag_probs[frag] * map_hessian(center)
        V_f = frag_probs[frag] * map_gradient(center)
        H = H + H_f
        V = V + V_f
    return (H, V)


def map_gradient(center, W, f):
    deriv = np.matmul(W[f,:]) * softmax_derivative(x=center)
    return deriv / np.matmul(W[f,:], softmax(x=center))

def map_hessian(center, W, f):
    # TODO

def softmax_derivative(x)
    s = softmax(x)
    N = len(s)
    sbar = 1 - s
    return s * np.transpose(sbar) - (np.ones((N, N)) - np.eye(N)) * s[:, None]

def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)
