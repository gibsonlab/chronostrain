"""
    random.py
    Contains helpers for random sampling.
"""
import numpy as np


def choice_vectorized(p: np.array, axis=1, dtype=np.int):
    """
    Vectorized operation for sampling from many categorical distributions.
    https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
    :param p: a 2-D list of probabilities.
    :param axis: The axis over which the input `p` is stochastic. (Default: 1)
        E.g. if p is 2-d and each row sums to zero (e.g. sum over columns), use `axis=1`.
    :param dtype: The dtype to use to allocate resulting array of choices.
    :return:
    """
    r = np.expand_dims(
        np.random.rand(p.shape[1-axis]),
        axis=axis
    )
    result_arr = np.zeros(shape=1-axis, dtype=dtype)

    return np.argmax(
        p.cumsum(axis=axis) > r,
        axis=axis,
        out=result_arr
    )
