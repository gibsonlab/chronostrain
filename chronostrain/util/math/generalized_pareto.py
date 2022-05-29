from typing import Tuple
import numpy as np


def fit_pareto(x: np.ndarray) -> Tuple[float, float]:
    """
    Implementation of Zhang and Stephens (2009) for fitting a generalized Pareto distribution.

    :param x: the data to fit across.
    """
    n = len(x)
    x = np.sort(x)
    m = 20 + int(np.sqrt(n))
    def lx(b_, x_):
        b_ = np.expand_dims(b_, axis=1)  # B x 1
        x_ = np.expand_dims(x_, axis=0)  # 1 x X
        k_ = -np.mean(np.log(1 - b_ * x_), axis=1)
        return np.log(b_ / k_) + k_ - 1

    b = (1 / x[-1]) + (1 - np.sqrt(m / (np.arange(1, m+1) - 0.5))) / 3 / x[int((n/4) + .5) - 1]  # length m
    L = n * lx(b, x)

    w = np.exp(L)
    w = w / np.sum(w)

    b = np.sum(b * w)
    k_est = float(np.mean(np.log(1 - b * x)))  # note: original paper has this sign flipped by convention.
    sigma_est = float(k / b)
    return k_est, sigma_est


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import genpareto

    cut_point = 5
    k = 2.0
    sigma = 1.0

    num_samples = 1000
    samples = genpareto.rvs(size=num_samples, c=k, loc=cut_point, scale=sigma)
    plt.hist(samples)
    plt.show()
