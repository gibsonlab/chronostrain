"""
    numpy_helpers.py
    Contains efficient helper functions for useful operations.
"""
from typing import Union

from numba import njit
import numpy as np


def choice_vectorized(p: np.ndarray, axis=1, dtype=np.int) -> np.ndarray:
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


@njit
def first_occurrence_of(array, item):
    """
    Return the index of the first occurrence of `item` in `array`. If no instance is found, return None.
    :param array:
    :param item:
    :return:
    """
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx


@njit
def first_nonoccurrence_of(array: np.ndarray, item: Union[float, int]) -> int:
    """
    Return the index of the first NON-occurrence of `item` in `array`. If no instance is found, raise an error.
    :param array:
    :param item:
    :return:
    """
    for idx, val in enumerate(array):
        if val != item:
            return idx
    raise RuntimeError()


@njit
def last_nonoccurrence_of(array: np.ndarray, item: Union[float, int]) -> int:
    """
    Return the index of the last NON-occurrence of `item` in `array`. If no instance is found, raise an error.
    :param array:
    :param item:
    :return:
    """
    return len(array) - 1 - first_nonoccurrence_of(np.flip(array), item)
