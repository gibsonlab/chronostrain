import math
from typing import List

import numpy as np
import numba
import numba.cuda
from numba.typed import List as nList
import torch
from chronostrain.util.sparse import ColumnSectionedSparseMatrix, RowSectionedSparseMatrix, SparseMatrix


@torch.jit.script
def outer_sum(x: torch.Tensor, y: torch.Tensor, target_idx: int) -> torch.Tensor:
    return x.select(1, target_idx).view(-1, 1) + y.select(0, target_idx).view(1, -1)


@torch.jit.script
def log_rowscale_exp_helper(x_values: torch.Tensor,
                            x_locs_per_row: List[torch.Tensor],
                            y: torch.Tensor,
                            buffer: torch.Tensor):
    for row_idx in torch.arange(0, len(x_locs_per_row), 1):
        target_locs = x_locs_per_row[row_idx]
        buffer[target_locs] = x_values[target_locs] + y[row_idx]


def log_rowscale_exp(x: RowSectionedSparseMatrix, y: torch.Tensor) -> RowSectionedSparseMatrix:
    """
    Scales each row of exp(x) by exp(y), assuming that empty entries of x are -inf. Outputs the logarithm,
    as in logsumexp.

    :param x: The sparse matrix of logarithm values.
    :param y: A 1-d tensor of the logarithm of scaling factors.
    """
    value_buffer = torch.empty(x.values.size(), dtype=x.values.dtype, device=x.values.device)
    log_rowscale_exp_helper(
        x.values,
        x.locs_per_row,
        y,
        value_buffer
    )
    return RowSectionedSparseMatrix(
        indices=x.indices,
        values=value_buffer,
        dims=(x.rows, x.columns),
        force_coalesce=False,
        _explicit_locs_per_row=x.locs_per_row
    )


@torch.jit.script
def logsumexp_sparse_helper(
        x_values: torch.Tensor,
        x_rows: int,
        x_locs_per_row: List[torch.Tensor],
        buffer: torch.Tensor
):
    for r in torch.arange(0, x_rows, 1):
        buffer[r] = torch.logsumexp(x_values[x_locs_per_row[r]], dim=0).item()


def logsumexp_row_sparse(x: RowSectionedSparseMatrix) -> torch.Tensor:
    # preallocation to speed up cuda
    buffer = torch.empty(x.rows, device=x.values.device, dtype=x.values.dtype)

    logsumexp_sparse_helper(
        x.values,
        x.rows,
        x.locs_per_row,
        buffer
    )

    return buffer


@torch.jit.script
def log_mm_exp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes log(exp(X) @ exp(Y)) in a numerically stable, where log/exp are entrywise operations.
    This implementation uses the identity (u|v) @ (x|y)^T = ux^T + vy^T, and uses a column-partitioning recursive
    strategy of depth log(n) to maintain O(mp) memory usage, where X is (m x n) and Y is (n x p).
    Ultimately, the formula is logsumexp(u + x^T, v + y^T), where logsumexp is applied entrywise and the plus(+) is an
    outer sum.

    Is optimized for a column-major x and a row-major y (e.g. x is the transpose of a natively created tensor()).
    """
    assert x.shape[1] == y.shape[0]
    assert x.shape[1] > 0

    ans = outer_sum(x, y, 0)
    for target_idx in range(1, x.shape[1]):
        next_sum = outer_sum(x, y, target_idx)
        ans = torch.logsumexp(
            torch.stack([ans, next_sum], dim=0),
            dim=0
        )
    return ans


@torch.jit.script
def spdense_outer_sum(x_indices: torch.Tensor,
                      x_values: torch.Tensor,
                      x_rows: int,
                      x_locs_per_col: List[torch.Tensor],
                      y: torch.Tensor,
                      target_idx: int) -> torch.Tensor:
    nz_targets = x_locs_per_col[target_idx]  # Extract the location of the nonzero elements from x.
    target_rows = x_indices[0, nz_targets]  # Get the relevant row indices.

    z = -float('inf') * torch.ones(
        x_rows, y.shape[1],
        device=x_values.device,
        dtype=x_values.dtype
    )

    z[target_rows, :] = x_values[nz_targets].view(-1, 1) + y[target_idx, :].view(1, -1)
    return z


@torch.jit.script
def log_mm_exp_spdense_helper(x_indices: torch.Tensor,
                              x_values: torch.Tensor,
                              x_rows: int,
                              x_cols: int,
                              x_locs_per_col: List[torch.Tensor],
                              y: torch.Tensor) -> torch.Tensor:
    """
    The jit-compiled version of log_spmm_exp which uses torch.logsumexp.
    """
    assert x_cols == y.shape[0]

    ans = spdense_outer_sum(x_indices, x_values, x_rows, x_locs_per_col, y, 0)
    for target_idx in range(1, x_cols):
        next_sum = spdense_outer_sum(x_indices, x_values, x_rows, x_locs_per_col, y, target_idx)
        ans = torch.logsumexp(
            torch.stack([ans, next_sum], dim=0),
            dim=0
        )
    return ans


def log_mm_exp_spdense(x: ColumnSectionedSparseMatrix, y: torch.Tensor) -> torch.Tensor:
    """
    Computes log(exp(X) @ exp(Y)) in a numerically stable, where log/exp are entrywise operations.
    """
    # TODO: optimize this slightly to boost BBVI cuda performance.
    return log_mm_exp_spdense_helper(
        x.indices, x.values, x.rows, x.columns, x.locs_per_column, y
    )


@torch.jit.script
def densesp_outer_sum(x: torch.Tensor,
                      y_indices: torch.Tensor,
                      y_values: torch.Tensor,
                      y_cols: int,
                      y_locs_per_row: List[torch.Tensor],
                      target_idx: int) -> torch.Tensor:
    """
    Given a target index k, computes the k-th summand of the dot product <u,v> = \SUM_k u_k v_k,
    for each row u of x, and each column v of y.

    Note that k here specifies a column of x, and a row of y, hence y is required to be the specification of a
    row-sliceable matrix.
    """
    nz_targets = y_locs_per_row[target_idx]  # Extract the location of the nonzero elements from x.
    target_cols = y_indices[0, nz_targets]  # Get the relevant column indices.

    z = -float('inf') * torch.ones(
        x.shape[0], y_cols,
        device=y_values.device,
        dtype=y_values.dtype
    )

    z[:, target_cols] = x[:, target_idx].view(-1, 1) + y_values[nz_targets].view(1, -1)
    return z


@torch.jit.script
def log_mm_exp_densesp_helper(x: torch.Tensor,
                        y_indices: torch.Tensor,
                        y_values: torch.Tensor,
                        y_rows: int,
                        y_cols: int,
                        y_locs_per_row: List[torch.Tensor]) -> torch.Tensor:
    """
    The jit-compiled version of log_spmm_exp which uses torch.logsumexp.
    """
    assert x.shape[1] == y_rows

    ans = densesp_outer_sum(x, y_indices, y_values, y_cols, y_locs_per_row, 0)
    # ans = sp_outer_sum(x_indices, x_values, x_rows, x_locs_per_col, y, 0)
    for target_idx in range(1, y_rows):
        next_sum = densesp_outer_sum(x, y_indices, y_values, y_cols, y_locs_per_row, target_idx)
        ans = torch.logsumexp(
            torch.stack([ans, next_sum], dim=0),
            dim=0
        )
    return ans


def log_mm_exp_densesp(x: torch.Tensor, y: RowSectionedSparseMatrix) -> torch.Tensor:
    """
    Computes log(exp(X) @ exp(Y)) in a numerically stable, where log/exp are entrywise operations.
    """
    return log_mm_exp_densesp_helper(
        x, y.indices, y.values, y.rows, y.columns, y.locs_per_row
    )


@numba.njit
def meshgrid2d(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for i in range(x.size):
        for j in range(y.size):
            xx[i, j] = x[i]
            yy[i, j] = y[j]
    return xx.flatten(), yy.flatten()


def log_spspmm_exp_helper(x_indices,
                          x_values,
                          x_locs_per_col,
                          y_indices,
                          y_values,
                          y_locs_per_row,
                          ans_buffer):
    for l_idx in range(len(x_locs_per_col)):
        target_x_locs = x_locs_per_col[l_idx]
        target_y_locs = y_locs_per_row[l_idx]
        for x_loc in target_x_locs:
            for y_loc in target_y_locs:
                r = x_indices[0, x_loc]
                c = y_indices[1, y_loc]

                A = ans_buffer[r, c]
                B = x_values[x_loc] + y_values[y_loc]
                C = max(A, B)
                ans_buffer[r, c] = C + math.log(math.exp(A - C) + math.exp(B - C))
                # ans_buffer[r, c] = np.logaddexp(, x_values[x_loc] + y_values[y_loc])


log_spspmm_exp_helper_jit = numba.jit(log_spspmm_exp_helper, nopython=True)
log_spspmm_exp_helper_cuda = numba.cuda.jit(log_spspmm_exp_helper)


def log_spspmm_exp(x: ColumnSectionedSparseMatrix, y: RowSectionedSparseMatrix) -> torch.Tensor:
    """
    Computes log(exp(X) @ exp(Y)) in a numerically stable, where log/exp are entrywise operations.
    Unlike log_mm_exp, X is given as a sparse matrix (empty entries are assumed to be -inf).
    This implementation uses the identity (A + B)C = AC + BC, and uses a col-partitioning recursive strategy
    of depth log(n) to maintain O(mp) memory usage, where X is (m x n) and Y is (n x p).
    """
    if x.columns != y.rows:
        raise ValueError("Columns of x and rows of y must match. Instead got ({} x {}) @ ({} x {})".format(
            x.rows, x.columns, y.rows, y.columns
        ))

    if x.rows == 0:
        return torch.empty(0, y.columns, dtype=y.values.dtype, device=y.values.device)
    elif y.columns == 0:
        return torch.empty(x.rows, 0, dtype=x.values.dtype, device=y.values.device)
    elif x.columns == 0 and y.rows == 0:
        raise RuntimeError("Cannot apply operation to ({} x 0) @ (0 x {}) matrix.".format(
            x.rows, y.columns
        ))

    """
    https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
    """
    ans_buffer = np.full((x.rows, y.columns), np.NINF, float)

    # TODO: do this directly on CUDA. (See note below)
    """
    Previously, ran into an issue where x_locs_per_row/y_locs_per_col (A list of
    CUDA arrays) was not indexable. One needs to pre-allocate a large array, or find a different representation
    in memory.
    """
    typed_x_locs = nList()
    for loc in x.locs_per_column:
        typed_x_locs.append(loc.cpu().numpy())
    typed_y_locs = nList()
    for loc in y.locs_per_row:
        typed_y_locs.append(loc.cpu().numpy())
    log_spspmm_exp_helper_jit(
        x.indices.cpu().numpy(),
        x.values.cpu().numpy(),
        typed_x_locs,
        y.indices.cpu().numpy(),
        y.values.cpu().numpy(),
        typed_y_locs,
        ans_buffer
    )

    # Call jit-enabled helper
    return torch.tensor(
        ans_buffer,
        dtype=x.values.dtype,
        device=x.values.device
    )
