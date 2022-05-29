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
        k_ = -np.mean(np.log(1 - b_ * x_))
        return np.log(b_ / k_) + k_ - 1

    b = (1 / x[-1]) + (1 - np.sqrt(m / (np.arange(1, m+1) - 0.5))) / 3 / x[int((n/4) + .5) - 1]  # length m
    w = np.copy(b)
    L = n * lx(
        b_=np.expand_dims(b, axis=1),
        x_=np.expand_dims(x, axis=0)
    )
