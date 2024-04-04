from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as np
import jax.experimental.sparse as jsparse
import numpy as cnp


def sparse_matrix_paths(base_path: Path) -> Tuple[Path, Path, Path]:
    return (
        base_path.parent / f'{base_path.stem}.indices.npy',
        base_path.parent / f'{base_path.stem}.data.npy',
        base_path.parent / f'{base_path.stem}.shape.npy'
    )


def save_sparse_matrix(path: Path, matrix: jsparse.BCOO):
    # Split up the files; workaround for jax.numpy.savez not retaining proper dtypes for bfloat16.
    p1, p2, p3 = sparse_matrix_paths(path)
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


# @numba.njit
# def _log_spspmm_exp_numba(x_indices: cnp.ndarray, x_values: cnp.ndarray,
#                           y_indices: cnp.ndarray, y_values: cnp.ndarray,
#                           z: np.ndarray):
#     for _b in range(len(y_values)):
#         k = y_indices[_b, 0]
#         tgt_y_col = y_indices[_b, 1]
#         tgt_y_val = y_values[_b]
#
#         tgt_x_locs = x_indices[:, 1] == k
#         tgt_x_vals = x_values[tgt_x_locs]
#         tgt_x_rows = x_indices[tgt_x_locs, 0]
#
#         newvals = tgt_x_vals + tgt_y_val
#         z[tgt_x_rows, tgt_y_col] = cnp.logaddexp(z[tgt_x_rows, tgt_y_col], newvals)
#
#
# def log_spspmm_exp(x: jsparse.BCOO, y: jsparse.BCOO):
#     z = cnp.full(shape=(x.shape[0], y.shape[1]), fill_value=-cnp.inf)
#     _log_spspmm_exp_numba(
#         cnp.array(x.indices), cnp.array(x.read_frags),
#         cnp.array(y.indices), cnp.array(y.read_frags),
#         z
#     )
#     return np.array(z)


def _log_row_scale(scale, tgt_row, y_indices, y_values, answer_buf):
    """
    Given a scalar (scale), extract and scale the k-th row (tgt_row) of the matrix Y, given by a COO specification (indices, values).
    """
    v = np.where(
        y_indices[:, 0] == tgt_row,
        y_values + scale,
        -cnp.inf
    )
    return answer_buf.at[y_indices[:, 1]].max(v)


def _log_col_scale(scale, tgt_col, x_indices, x_values, answer_buf):
    """
    Given a scalar (scale), extract and scale the k-th column (tgt_col) of the matrix X, given by a COO specification (indices, values).
    """
    v = np.where(
        x_indices[:, 1] == tgt_col,
        x_values + scale,
        -cnp.inf
    )
    return answer_buf.at[x_indices[:, 0]].max(v)


def _log_spspmm_exp_lax_sparsey(x_indices: np.ndarray, x_values: np.ndarray,
                        y_indices: np.ndarray, y_values: np.ndarray,
                        ans_buf: np.ndarray) -> np.ndarray:
    """
    jax.lax specific implementation for JIT compilation.
    Strategy: Break down X @ Y = \SUM_{ij} X @ Y_ij where Y_ij is the matrix whose entries are all zero, except the ij-th entry equal to Y[i,j].
    Implementation:
    - Iterate through each element of y (at COO location [i, j]), treat it as an instance of _log_row_scale.
    - Combine all answers using logaddexp.
    """
    def _helper(i, carry):
        _z, _x_indices, _x_values, _y_indices, _y_values = carry
        y_row = _y_indices[i, 0]
        y_col = _y_indices[i, 1]
        y_val = _y_values[i]
        new_col = _log_col_scale(y_val, y_row, x_indices, x_values, np.full(shape=_z.shape[0], fill_value=-cnp.inf))
        return (
            _z.at[:, y_col].set(np.logaddexp(_z[:, y_col], new_col)),
            _x_indices, _x_values, _y_indices, _y_values
        )

    return jax.lax.fori_loop(
        0, len(y_values),
        _helper,
        (ans_buf, x_indices, x_values, y_indices, y_values)
    )[0]


def _log_spspmm_exp_lax_sparsex(x_indices: np.ndarray, x_values: np.ndarray,
                        y_indices: np.ndarray, y_values: np.ndarray,
                        ans_buf: np.ndarray) -> np.ndarray:
    """
    jax.lax specific implementation for JIT compilation.
    Strategy: Break down X @ Y = \SUM_{ij} (X_ij @ Y), where X_ij is the matrix whose entries are all zero, except the ij-th entry equal to X[i,j].
    Implementation:
    - Iterate through each element of x (at COO location [i, j]), treat it as an instance of _log_row_scale.
    - Combine all answers using logaddexp.
    """
    def _helper(i, carry):
        _z, _x_indices, _x_values, _y_indices, _y_values = carry
        x_row = _x_indices[i, 0]
        x_col = _x_indices[i, 1]
        x_val = _x_values[i]
        new_row = _log_row_scale(x_val, x_col, _y_indices, _y_values, np.full(shape=_z.shape[1], fill_value=-cnp.inf))
        return (
            _z.at[x_row].set(np.logaddexp(_z[x_row], new_row)),
            _x_indices, _x_values, _y_indices, _y_values
        )

    return jax.lax.fori_loop(
        0, len(x_values),
        _helper,
        (ans_buf, x_indices, x_values, y_indices, y_values)
    )[0]


_thresh_count = 1e6
def log_spspmm_exp(x: jsparse.BCOO, y: jsparse.BCOO):
    """
    same idea as log_spspmm_exp, but assumes one is far sparser than the other.
    """
    if len(x.data) < len(y.data):
        if len(x.data) < _thresh_count:
            return _log_spspmm_exp_lax_sparsex(
                x.indices, x.data,
                y.indices, y.data,
                np.full(shape=(x.shape[0], y.shape[1]), fill_value=-cnp.inf)
            )
        else:
            print(f"Encountered BCOO with nnz>{_thresh_count}. Using experimental matmul code.")
            return log_spspmm_exp_experimental(x, y)
    else:
        if len(y.data) < _thresh_count:
            return _log_spspmm_exp_lax_sparsey(
                x.indices, x.data,
                y.indices, y.data,
                np.full(shape=(x.shape[0], y.shape[1]), fill_value=-cnp.inf)
            )
        else:
            print(f"Encountered BCOO with nnz>{_thresh_count}. Using experimental matmul code.")
            return log_spspmm_exp_experimental(x, y)


def log_spspmm_exp_experimental(x: jsparse.BCOO, y: jsparse.BCOO):
    """
    Uses the idea from http://gcucurull.github.io/deep-learning/2020/06/03/jax-sparse-matrix-multiplication/, where we
    take advantage of a dense structure and use jax/lax native take+segment_sum calls.
    :param x: a sparse BCOO matrix, whose rows will be densified.
    :param y: a sparse BCOO matrix
    :return:
    """
    n_rows = x.shape[0]
    return np.stack([
        sp_vecmat_logaddexp(x[i, :], y)
        for i in range(n_rows)
    ], axis=0)


def sp_vecmat_logaddexp(x: jsparse.BCOO, y: jsparse.BCOO):
    """
    Evaluates x @ y, assuming x is a row vector (1-d array representation) and y is a sparse BCOO matrix.
    The inner product is defined as <x,y> = logsumexp(x+y), empty entries represent -inf = log(0), not zero.
    Outputs a 1-d dense row vector.
    """
    assert x.ndim == 1
    x = jsparse.BCOO.todense(x)
    x = x.at[x == 0].set(-np.inf)

    indexes = y.indices  # shape is (nnz, 2)
    rows = indexes[:, 0]
    cols = indexes[:, 1]

    values = y.data  # shape is (nnz,)
    _sum = x.take(rows) + values  # entrywise sum of the logarithms of the entries
    offsets = jax.ops.segment_max(_sum, cols, y.shape[1])

    return np.log(
        jax.ops.segment_sum(
            np.exp(
                # sum with offsets; entry is nan when evaluating (-inf) - (-inf), but it suffices to set these
                # differences equal to zero (since we will be re-adding -inf at the last step).
                np.nan_to_num(
                    _sum - offsets.take(cols),
                    copy=False,
                    nan=0.,  # rectify the NaNs.
                    posinf=np.inf,  # keep +/- inf the same.
                    neginf=-np.inf
                )
            ),
            cols,
            y.shape[1]
        )
    ) + offsets


def densesp_mm(x: np.ndarray, y: jsparse.BCOO) -> np.ndarray:
    raise NotImplementedError("TODO later")
