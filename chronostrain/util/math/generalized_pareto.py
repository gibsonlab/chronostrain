"""
Copyright 2017 Aki Vehtari, Tuomas Sivula
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. """


import numpy as np


def fit_pareto(x: np.ndarray,
               loc: float = 0.0):
    """Estimate the paramaters for the Generalized Pareto Distribution (GPD)
    Returns empirical Bayes estimate for the parameters of the two-parameter
    generalized Parato distribution given the data.
    Parameters
    ----------
    x : ndarray
        One dimensional data array
    loc: float

    Returns
    -------
    k, sigma : float
        estimated parameter values
    Notes
    -----
    This function returns a negative of Zhang and Stephens's k, because it is
    more common parameterisation.
    """
    if x.ndim != 1 or len(x) <= 1:
        raise ValueError("Invalid input array.")

    x = np.copy(x) - loc
    sort = np.argsort(x)
    n = len(x)
    PRIOR = 3
    m = 30 + int(np.sqrt(n))

    bs = np.arange(1, m + 1, dtype=float)
    bs -= 0.5
    np.divide(m, bs, out=bs)
    np.sqrt(bs, out=bs)
    np.subtract(1, bs, out=bs)

    bs /= PRIOR * x[sort[int(n/4 + 0.5) - 1]]
    bs += 1 / x[sort[-1]]

    ks = np.negative(bs)
    temp = ks[:, None] * x
    np.log1p(temp, out=temp)
    np.mean(temp, axis=1, out=ks)

    L = bs / ks
    np.negative(L, out=L)
    np.log(L, out=L)
    L -= ks
    L -= 1
    L *= n

    temp = L - L[:, None]
    np.exp(temp, out=temp)
    w = np.sum(temp, axis=1)
    np.divide(1, w, out=w)

    # remove negligible weights
    dii = w >= 10 * np.finfo(float).eps
    if not np.all(dii):
        w = w[dii]
        bs = bs[dii]
    # normalise w
    w /= w.sum()

    # posterior mean for b
    b = np.sum(bs * w)
    # Estimate for k, note that we return a negative of Zhang and
    # Stephens's k, because it is more common parameterisation.
    temp = (-b) * x
    np.log1p(temp, out=temp)
    k = np.mean(temp)

    # estimate for sigma
    sigma = -k / b * n / (n - 0)

    # weakly informative prior for k
    a = 10
    k = k * n / (n+a) + a * 0.5 / (n+a)
    return k, sigma



# from typing import Tuple
#
# import numpy as np
# import scipy.special
#
#
# def fit_pareto(x: np.ndarray, loc: float) -> Tuple[float, float]:
#     """
#     Implementation of Zhang and Stephens (2009) for fitting a generalized Pareto distribution.
#
#     :param x: the data to fit across.
#     """
#     n = len(x)
#     x = np.sort(x) - loc
#     m = 20 + int(np.sqrt(n))
#
#     def lx(b_: np.ndarray, x_: np.ndarray) -> np.ndarray:
#         k_ = -np.mean(np.log(1 - np.expand_dims(b_, axis=1) * np.expand_dims(x_, axis=0)), axis=1)
#         return np.log(b_ / k_) + k_ - 1
#
#     b = (1 / x[-1]) + (1 - np.sqrt(m / (np.arange(1, m+1) - 0.5))) / 3 / x[int((n/4) + .5) - 1]  # length m
#     L = n * lx(b, x)
#     w = np.exp(L - scipy.special.logsumexp(L))
#
#     b = np.sum(b * w)
#     k_est = float(np.mean(np.log(1 - b * x)))  # note: original paper has this sign flipped by convention.
#     sigma_est = float(-k_est / b)
#     return k_est, sigma_est
#
#
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from scipy.stats import genpareto
#
#     cut_point = 5
#     k = 0.2
#     sigma = 50.0
#
#     num_samples = 10000
#     true_dist = genpareto(c=k, loc=cut_point, scale=sigma)
#     samples = true_dist.rvs(size=num_samples)
#
#     k_hat, sigma_hat = fit_pareto(samples, loc=np.min(samples))
#     est_dist = genpareto(c=k_hat, loc=np.min(samples), scale=sigma_hat)
#     print(f"k = {k}, sigma = {sigma}")
#     print(f"k_est = {k_hat}, sigma_est = {sigma_hat}")
#
#     x = np.linspace(0, 100, 10000)
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#     ax.plot(x, true_dist.pdf(x))
#     ax.plot(x, est_dist.pdf(x))
#     plt.show()
