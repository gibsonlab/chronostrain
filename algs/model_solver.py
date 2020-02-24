import numpy as np

# ===============
# An implementation of the proposed algorithms.
# ===============

class GenerativeModel:
    def __init__(self, times, mu, tau, W, strains):
        self.times = times
        self.mu = mu
        self.tau = tau
        self.W = W
        self.strains = strains


def variational_learn(model, reads):

    # Initialization
    means = [np.zeros(model.strains, 1) for t in range(model.times)]

    raise NotImplementedError()