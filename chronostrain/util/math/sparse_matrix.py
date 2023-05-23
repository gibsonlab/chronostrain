from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as np
import jax.experimental.sparse as jsparse
import numpy as cnp
import numba


def sparse_matrix_paths(base_path: Path) -> Tuple[Path, Path, Path]:
    return (
        base_path.parent / f'{base_path.stem}.indices.npy',
        base_path.parent / f'{base_path.stem}.data.npy',
        base_path.parent / f'{base_path.stem}.shape.npy'
    )


def save_sparse_matrix(path: Path, matrix: jsparse.BCOO):
    p1, p2, p3 = sparse_matrix_paths(path)  # Workaround for jax.numpy.savez not retaining proper dtypes for bfloat16.
    path.touch()
    np.save(str(p1), matrix.indices)
    np.save(str(p2), matrix.data)
    np.save(str(p3), matrix.shape)


def load_sparse_matrix(path: Path) -> jsparse.BCOO:
    p1, p2, p3 = sparse_matrix_paths(path)
    indices = np.load(p1)
    data = np.load(p2)
    shape = cnp.load(str(p3)).astype(int).tolist()
    return jsparse.BCOO(
        (data, indices), shape=shape
    )


def scale_row(matrix: jsparse.BCOO, vec: np.ndarray) -> jsparse.BCOO:
    """
    Scales each row of the matrix by the value specified in scales. matrix.shape[0] and len(scales) must match.
    This is not an in-place operation.
    :param matrix:
    :param vec:
    :return: A new sparse matrix with the specified row-scaling.
    """
    return jsparse.BCOO(
        (matrix.data * vec[matrix.indices[:, 0]], matrix.indices),
        shape=matrix.shape
    )


def column_normed_row_sum(x: jsparse.BCOO) -> np.ndarray:
    """
    Normalize each column (so that each column sums to 1 (sum, dim=0)), and then sum over the rows (sum, dim=1).

    :return: A dense 1-d vector.
    """
    column_sums = jsparse.bcoo_reduce_sum(x, axes=[0]).todense()
    rescaled_values = x.data / column_sums[x.indices[:, 1]]
    return jsparse.bcoo_reduce_sum(
            jsparse.BCOO(
            (rescaled_values, x.indices),
            shape=x.shape
        ),
        axes=[1]
    ).todense()


@numba.jit(nopython=False)
def _log_spspmm_exp_numba(x_indices: cnp.ndarray, x_values: cnp.ndarray,
                          y_indices: cnp.ndarray, y_values: cnp.ndarray,
                          z: np.ndarray):
    for _b in range(len(y_values)):
        k = y_indices[_b, 0]
        tgt_y_col = y_indices[_b, 1]
        tgt_y_val = y_values[_b]

        tgt_x_locs = x_indices[:, 1] == k
        tgt_x_vals = x_values[tgt_x_locs]
        tgt_x_rows = x_indices[tgt_x_locs, 0]

        newvals = tgt_x_vals + tgt_y_val
        z[tgt_x_rows, tgt_y_col] = cnp.logaddexp(
            z[tgt_x_rows, tgt_y_col], newvals
        )


def log_spspmm_exp_sparsey(x: jsparse.BCOO, y: jsparse.BCOO):
    z = cnp.full(shape=(x.shape[0], y.shape[1]), fill_value=-cnp.inf)
    _log_spspmm_exp_numba(
        cnp.array(x.indices), cnp.array(x.data),
        cnp.array(y.indices), cnp.array(y.data),
        z
    )
    return np.array(z)


# @jax.jit
# def _log_row_scale(x_val, tgt_row, y_indices, y_values, answer_buf):
#     """
#     Given a scalar (x_val), extract and scale the k-th row (tgt_row) of the matrix Y, given by a COO specification (indices, values).
#     """
#     v = np.where(
#         y_indices[:, 0] == tgt_row,
#         y_values + x_val,
#         -cnp.inf
#     )
#     return answer_buf.at[y_indices[:, 1]].max(v)
#
#
# @jax.jit
# def _log_spspmm_exp_lax(x_indices: np.ndarray, x_values: np.ndarray,
#                         y_indices: np.ndarray, y_values: np.ndarray,
#                         ans_buf: np.ndarray) -> np.ndarray:
#     """
#     jax.lax specific implementation for JIT compilation.
#     Strategy: Break down X @ Y = \SUM_{ij} (X_ij @ Y), where X_ij is the matrix whose entries are all zero, except the ij-th entry equal to X[i,j].
#     Implementation:
#     - Iterate through each element of x (at COO location [i, j]), treat it as an instance of _log_row_scale.
#     - Combine all answers using logaddexp.
#     """
#
#     def _helper(i, z):
#         x_row = x_indices[i, 0]
#         x_col = x_indices[i, 1]
#         x_val = x_values[i]
#         new_row = _log_row_scale(x_val, x_col, y_indices, y_values,
#                                  np.full(shape=ans_buf.shape[1], fill_value=-cnp.inf))
#         return z.at[x_row].set(
#             np.logaddexp(z[x_row], new_row)
#         )
#
#     return jax.lax.fori_loop(
#         0, len(x_values),
#         _helper,
#         ans_buf
#     )
#
#
# def log_spspmm_exp_sparsey(x: jsparse.BCOO, y: jsparse.BCOO):
#     """
#     same idea as log_spspmm_exp, but assumes y is far sparser than x. Tries to use this to leverage an easier calculation.
#     """
#     return _log_spspmm_exp_lax(
#         x.indices, x.data,
#         y.indices, y.data,
#         np.full(shape=(x.shape[0], y.shape[1]), fill_value=-cnp.inf)
#     )


@jax.jit
def densesp_mm(x: np.ndarray, y: jsparse.BCOO) -> np.ndarray:
    raise NotImplementedError("TODO later")
