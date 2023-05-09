import jax
import jax.numpy as np

import numpy as cnp
_NORMAL_LOG_FACTOR = cnp.log(2 * cnp.pi)

@jax.jit
def tril_matrix_of(tril_weights: np.ndarray, diag_weights: np.ndarray):
    d = len(diag_weights)
    idx = cnp.tril_indices(d, k=-1)
    return np.zeros((d, d), dtype=tril_weights.dtype).at[idx].set(tril_weights)


# @jax.jit
def tril_linear_transform_with_bias(
        tril_weights: np.ndarray,
        diag_weights: np.ndarray,
        bias: np.ndarray,
        x: np.ndarray
):
    """
    :param tril_weights: (N x (N-1) / 2)-length array. (lower triangular elements)
    :param diag_weights: length N array (log-diagonal elements)
    :param bias: length N array (vector 'b')
    :param x: (M x N) array
    :return: Computes x @ A.T + b, where A is the lower-triangular matrix specified by the weights.
    """
    return np.matmul(
        x,
        tril_matrix_of(tril_weights, diag_weights).T + np.diag(diag_weights)
    ) + bias


@jax.jit
def tril_linear_transform_no_bias(
        tril_weights: np.ndarray,
        diag_weights: np.ndarray,
        x: np.ndarray
):
    """
    :param tril_weights: (N x (N-1) / 2)-length array. (lower triangular elements)
    :param diag_weights: length N array (log-diagonal elements)
    :param bias: length N array (vector 'b')
    :param x: (M x N) array
    :return: Computes x @ A.T + b, where A is the lower-triangular matrix specified by the weights.
    """
    return np.matmul(
        x,
        tril_matrix_of(tril_weights, diag_weights).T + np.diag(diag_weights)  # A.t is triu, so A is tril.
    )


@jax.jit
def gaussian_entropy(tril_weights: np.ndarray, diag_weights: np.ndarray):
    return np.log(diag_weights).sum()
