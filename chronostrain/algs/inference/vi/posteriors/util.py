import jax
import jax.numpy as np

import numpy as cnp
_NORMAL_LOG_FACTOR = cnp.log(2 * cnp.pi)

@jax.jit
def tril_matrix_of(tril_weights: np.ndarray, diag_weights: np.ndarray):
    n_features = len(diag_weights)
    A = np.zeros(shape=(n_features, n_features), dtype='float32')
    tril_r, tril_c = np.tril_indices(n_features, n_features, -1)
    diag_r = np.arange(0, n_features)
    A[tril_r, tril_c] = tril_weights
    A[diag_r, diag_r] = diag_weights
    return A


@jax.jit
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
        tril_matrix_of(tril_weights, diag_weights).T
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
        tril_matrix_of(tril_weights, diag_weights).T
    )


@jax.jit
def gaussian_entropy(tril_weights: np.ndarray, diag_weights: np.ndarray):
    n = len(diag_weights)
    half_log_det_cov = np.log(diag_weights).sum()
    return 0.5 * n * (1 + _NORMAL_LOG_FACTOR) + half_log_det_cov
