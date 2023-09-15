import jax
import jax.numpy as np

import numpy as cnp
_NORMAL_LOG_FACTOR = cnp.log(2 * cnp.pi)

@jax.jit
def tril_matrix_of(tril_weights: np.ndarray, diag_weights: np.ndarray):
    # Adapted from tensorflow's solution
    # https://github.com/google/jax/discussions/10146
    # https://github.com/tensorflow/probability/blob/913085acf56c19436ed59f863eb2985d973d90a5/tensorflow_probability/python/math/linalg.py#L780
    # n = len(diag_weights)
    # return np.tril(
    #     np.reshape(
    #         np.concatenate([tril_weights, tril_weights[n:][::-1]]),
    #         [n, n]
    #     ),
    #     k=-1
    # ) + np.diag(diag_weights)  # Erase the extra diagonal entries and replace it with the given diagonal weights.

    d = len(diag_weights)
    idx = cnp.tril_indices(d, k=-1)
    # idx = cnp.tril_indices(d, k=0)
    return np.zeros((d, d), dtype=tril_weights.dtype).at[idx].set(tril_weights) + np.diag(diag_weights)


@jax.jit
def tril_solve(
        tril_weights: np.ndarray,
        diag_weights: np.ndarray,
        b: np.ndarray
) -> np.ndarray:
    return jax.scipy.linalg.solve_triangular(
        tril_matrix_of(tril_weights, diag_weights),  # (DxD) lower triangular matrix A
        b.T,  # vector b, transpose turns it from (NxD) to (DxN), batch dim is N
        lower=True
    ).T  # solve Ax=b, batched into (DxN) --> (NxD) transposed


@jax.jit
def tril_linear_transform_with_bias(
        tril_weights: np.ndarray,
        diag_weights: np.ndarray,
        bias: np.ndarray,
        x: np.ndarray
) -> np.ndarray:
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
) -> np.ndarray:
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
def gaussian_entropy(tril_weights: np.ndarray, diag_weights: np.ndarray) -> np.ndarray:
    return np.log(diag_weights).sum()
