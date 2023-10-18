import numpy as np
from typing import Tuple
from statsmodels.discrete.discrete_model import NegativeBinomial


def negbin_fit(x: np.ndarray) -> Tuple[float, float]:
    """
    Fit a negative binomial distribution parametrized by N and p.
    :param x: the obsservations to fit a negbin for.
    :return: A tuple of floats (n_est, p_est).
    """
    model = NegativeBinomial(x, np.ones_like(x))
    results = model.fit()
    p = 1 / (1 + np.exp(results.params[0]) * results.params[1])
    n = np.exp(results.params[0]) * p / (1 - p)
    return n, p
