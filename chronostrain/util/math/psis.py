from typing import Tuple

import numpy as np
import scipy.stats, scipy.special

from .generalized_pareto import fit_pareto


def psis_smooth_ratios(log_raw_ratios: np.ndarray, k_min: float = 1/3) -> Tuple[np.ndarray, float]:
    """
    Based on citation:
    2019 Vehtari, Simpson, Gelman, Yao, Gabry. 'Pareto Smoothed Importance Sampling'

    Parts of code lifted from authors' original implementation.
    (https://github.com/avehtari/PSIS/blob/master/py/psis.py)

    :param log_raw_ratios:
    :param k_min: Reweighting does not happen if the empirical Pareto k-hat does not exceed this value.
    :return: A pair of values:
    1) Pareto-smoothed estimation (log-)weights, to be used in importance sampling calculation.
    2) the estimated Pareto distribution's empirical k (k-hat).
    """
    n = len(log_raw_ratios)
    if n <= 1:
        raise ValueError("More than one log-weight needed.")

    # allocate new array for output
    sort_idxs = np.argsort(log_raw_ratios)
    wts = np.copy(log_raw_ratios)

    # precalculate constants
    cutoff_idx = -int(np.ceil(min(0.2 * n, 3 * np.sqrt(n)))) - 1

    # divide log weights into body and right tail
    log_cutoff_value = max(
        wts[sort_idxs[cutoff_idx]],
        np.log(np.finfo(float).tiny)  # smallest possible value
    )
    cutoff_value = np.exp(log_cutoff_value)

    tailinds, = np.where(wts > log_cutoff_value)
    tail_sz = len(tailinds)

    if tail_sz <= 4:
        raise RuntimeError("Not enough tail samples to perform this estimate.")

    tail_wts = wts[tailinds]
    k_est, sigma_est = fit_pareto(np.exp(tail_wts), loc=cutoff_value)

    # Regularization from Appendix C of paper
    k_est = (tail_sz * k_est + 5) / (tail_sz + 10)

    # Only smooth if long-tailed.
    if k_est >= k_min:
        # compute ordered statistic for the fit
        sti = np.arange(0.5, tail_sz)
        sti /= tail_sz

        # Smoothing formula (inverse-cdf of gen. pareto)
        qq = scipy.stats.genpareto.ppf(
            sti,
            c=k_est,
            scale=sigma_est
        ) + cutoff_value
        np.log(qq, out=qq)

        # place the smoothed tail into the output array
        tail_ordering = np.argsort(tail_wts)
        wts[tailinds[tail_ordering]] = qq

        # truncate smoothed values to the largest raw (log-)weight 0
        wts[wts > 0] = 0

    # renormalize weights
    wts -= scipy.special.logsumexp(wts)
    return wts, k_est
