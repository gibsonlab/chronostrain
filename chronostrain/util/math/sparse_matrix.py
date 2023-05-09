from pathlib import Path

import jax
import jax.numpy as np
import jax.experimental.sparse as jsparse


def save_sparse_matrix(path: Path, matrix: jsparse.BCOO):
    np.savez(
        path,
        data=matrix.data,
        indices=matrix.indices,
        shape=matrix.shape
    )


def load_sparse_matrix(path: Path) -> jsparse.BCOO:
    f = np.load(path)
    return jsparse.BCOO(
        (f['data'], f['indices']), shape=list(f['shape'])
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


import numba
import numpy as cnp


# @numba.jit(nopython=False)
def _log_spspmm_exp_numba(x_indices: cnp.ndarray, x_values: cnp.ndarray,
                          y_indices: cnp.ndarray, y_values: cnp.ndarray,
                          x_rows: int, y_cols: int) -> cnp.ndarray:
    z = cnp.full(shape=(x_rows, y_cols), fill_value=-cnp.inf)

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
    return z


def log_spspmm_exp_sparsey(x: jsparse.BCOO, y: jsparse.BCOO):
    """
    same idea as log_spspmm_exp, but assumes y is far sparser than x. Tries to use this to leverage an easier calculation.
    """
    return np.array(
        _log_spspmm_exp_numba(
            cnp.array(x.indices), cnp.array(x.data),
            cnp.array(y.indices), cnp.array(y.data),
            x.shape[0], y.shape[1]
        )
    )


@jax.jit
def densesp_mm(x: np.ndarray, y: jsparse.BCOO) -> np.ndarray:
    raise NotImplementedError("TODO later")
