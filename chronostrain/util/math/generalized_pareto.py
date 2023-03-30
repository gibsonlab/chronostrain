from typing import Tuple

import numpy as np


def fit_pareto(x: np.ndarray, loc: float) -> Tuple[float, float]:
    """
    Implementation of Zhang and Stephens (2009) for fitting a generalized Pareto distribution.

    :param x: the data to fit across.
    """
    n = len(x)
    x = np.sort(x) - loc
    m = 20 + int(np.sqrt(n))

    def lx(b_: np.ndarray, x_: np.ndarray) -> np.ndarray:
        k_ = -np.mean(np.log1p(-np.expand_dims(b_, axis=1) * np.expand_dims(x_, axis=0)), axis=1)
        return np.log(b_ / k_) + k_ - 1

    b = (1 / x[-1]) + (1 - np.sqrt(m / (np.arange(1, m+1) - 0.5))) / 3 / x[int((n/4) + .5) - 1]  # length m
    L = n * lx(b, x)

    w = np.reciprocal(
        np.sum(
            np.exp(L - np.expand_dims(L, axis=1)),
            axis=1
        )
    )

    b = np.sum(b * w)
    k_est = float(np.mean(np.log1p(-b * x)))  # note: original paper has this sign flipped by convention.
    sigma_est = float(-k_est / b)

    # Regularization from Appendix C of paper
    k_est = (n * k_est + 5) / (n + 10)

    return k_est, sigma_est


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import genpareto

    cut_point = 5
    k = 0.1
    sigma = 3.0

    num_samples = 50
    true_dist = genpareto(c=k, loc=cut_point, scale=sigma)
    samples = true_dist.rvs(size=num_samples)

    k_hat, sigma_hat = fit_pareto(samples, loc=np.min(samples))
    est_dist = genpareto(c=k_hat, loc=np.min(samples), scale=sigma_hat)
    print(f"k = {k}, sigma = {sigma}")
    print(f"k_est = {k_hat}, sigma_est = {sigma_hat}")

    k_new, sigma_new = gpdfitnew(samples)
    print(f"k_new = {k_new}")

    x = np.linspace(0, 100, 10000)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(x, true_dist.pdf(x))
    ax.plot(x, est_dist.pdf(x))
    plt.show()
