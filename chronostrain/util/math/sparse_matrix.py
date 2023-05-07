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


@jax.jit
def log_spspmm_exp(x: jsparse.BCOO, y: jsparse.BCOO) -> np.ndarray:
    buf = []
    for i in range(x.shape[0]):
        buf_row = []
        for j in range(y.shape[1]):
            tmp = x[i] + y[:, j]
            buf_row.append(
                jax.scipy.special.logsumexp(tmp.todense())
            )
        buf.append(buf_row)
    return np.array(buf)


@jax.jit
def densesp_mm(x: np.ndarray, y: jsparse.BCOO) -> np.ndarray:
    raise NotImplementedError("TODO later")
