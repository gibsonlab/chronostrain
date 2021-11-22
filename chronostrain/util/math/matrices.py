from typing import List

import torch
from chronostrain.util.sparse import ColumnSectionedSparseMatrix, RowSectionedSparseMatrix


@torch.jit.script
def outer_sum(x: torch.Tensor, y: torch.Tensor, target_idx: int) -> torch.Tensor:
    return x.select(1, target_idx).view(-1, 1) + y.select(0, target_idx).view(1, -1)


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
def sp_outer_sum(x_indices: torch.Tensor,
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
def log_spmm_exp_helper(x_indices: torch.Tensor,
                        x_values: torch.Tensor,
                        x_rows: int,
                        x_cols: int,
                        x_locs_per_col: List[torch.Tensor],
                        y: torch.Tensor) -> torch.Tensor:
    """
    The jit-compiled version of log_spmm_exp.
    """
    assert x_cols == y.shape[0]

    ans = sp_outer_sum(x_indices, x_values, x_rows, x_locs_per_col, y, 0)
    for target_idx in range(1, x_cols):
        next_sum = sp_outer_sum(x_indices, x_values, x_rows, x_locs_per_col, y, target_idx)
        ans = torch.logsumexp(
            torch.stack([ans, next_sum], dim=0),
            dim=0
        )
    return ans


def log_spmm_exp(x: ColumnSectionedSparseMatrix, y: torch.Tensor) -> torch.Tensor:
    """
    Computes log(exp(X) @ exp(Y)) in a numerically stable, where log/exp are entrywise operations.
    Unlike log_mm_exp, X is given as a sparse matrix (empty entries are assumed to be -inf).
    This implementation uses the identity (A + B)C = AC + BC, and uses a col-partitioning recursive strategy
    of depth log(n) to maintain O(mp) memory usage, where X is (m x n) and Y is (n x p).
    """
    return log_spmm_exp_helper(
        x.indices, x.values, x.rows, x.columns, x.locs_per_column, y
    )


@torch.jit.script
def spsp_outer_sum(x_indices: torch.Tensor,
                   x_values: torch.Tensor,
                   x_rows: int,
                   x_target_locs: torch.Tensor,
                   y_indices: torch.Tensor,
                   y_values: torch.Tensor,
                   y_cols: int,
                   y_target_locs: torch.Tensor) -> torch.Tensor:
    x_target_rows = x_indices[0, x_target_locs]  # Get the relevant row indices.
    y_target_cols = y_indices[1, y_target_locs]

    z = -float('inf') * torch.ones(
        x_rows, y_cols,
        device=x_values.device,
        dtype=x_values.dtype
    )

    r, c = torch.meshgrid(x_target_rows, y_target_cols, indexing='ij')
    z[r, c] = x_values[x_target_locs].view(-1, 1) + y_values[y_target_locs].view(1, -1)
    return z


@torch.jit.script
def log_spspmm_exp_helper(x_indices: torch.Tensor,
                          x_values: torch.Tensor,
                          x_rows: int,
                          x_cols: int,
                          x_locs_per_col: List[torch.Tensor],
                          y_indices: torch.Tensor,
                          y_values: torch.Tensor,
                          y_rows: int,
                          y_cols: int,
                          y_locs_per_row: List[torch.Tensor]) -> torch.Tensor:
    assert x_cols == y_rows

    ans = spsp_outer_sum(
        x_indices,
        x_values,
        x_rows,
        x_locs_per_col[0],
        y_indices,
        y_values,
        y_cols,
        y_locs_per_row[0]
    )
    for target_idx in range(1, x_cols):
        next_sum = spsp_outer_sum(
            x_indices,
            x_values,
            x_rows,
            x_locs_per_col[target_idx],
            y_indices,
            y_values,
            y_cols,
            y_locs_per_row[target_idx],
        )
        ans = torch.logsumexp(
            torch.stack([ans, next_sum], dim=0),
            dim=0
        )
    return ans


def log_spspmm_exp(x: ColumnSectionedSparseMatrix, y: RowSectionedSparseMatrix) -> torch.Tensor:
    """
    Computes log(exp(X) @ exp(Y)) in a numerically stable, where log/exp are entrywise operations.
    Unlike log_mm_exp, X is given as a sparse matrix (empty entries are assumed to be -inf).
    This implementation uses the identity (A + B)C = AC + BC, and uses a col-partitioning recursive strategy
    of depth log(n) to maintain O(mp) memory usage, where X is (m x n) and Y is (n x p).
    """
    return log_spspmm_exp_helper(
        x.indices, x.values, x.rows, x.columns, x.locs_per_column,
        y.indices, y.values, y.rows, y.columns, y.locs_per_row
    )
