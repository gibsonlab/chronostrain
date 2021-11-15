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
                   x_locs_per_col: List[torch.Tensor],
                   y_indices: torch.Tensor,
                   y_values: torch.Tensor,
                   y_cols: int,
                   y_locs_per_row: List[torch.Tensor],
                   target_idx: int) -> torch.Tensor:
    x_nz_targets = x_locs_per_col[target_idx]  # Extract the location of the nonzero elements from x.
    x_target_rows = x_indices[0, x_nz_targets]  # Get the relevant row indices.

    y_nz_targets = y_locs_per_row[target_idx]
    y_target_cols = y_indices[y_nz_targets]

    z = -float('inf') * torch.ones(
        x_rows, y_cols,
        device=x_values.device,
        dtype=x_values.dtype
    )

    z[torch.meshgrid(x_target_rows, y_target_cols)] = x_values[x_nz_targets].view(-1, 1) + y_values[y_nz_targets].view(1, -1)
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
        x_locs_per_col,
        y_indices,
        y_values,
        y_cols,
        y_locs_per_row,
        0
    )
    for target_idx in range(1, x_cols):
        next_sum = spsp_outer_sum(
            x_indices,
            x_values,
            x_rows,
            x_locs_per_col,
            y_indices,
            y_values,
            y_cols,
            y_locs_per_row,
            target_idx
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



# @torch.jit.script
# def log_spmm_exp_helper(x_indices: torch.Tensor,
#                         x_values: torch.Tensor,
#                         x_rows: int,
#                         x_cols: int,
#                         y: torch.Tensor) -> torch.Tensor:
#     """
#     Computes log(exp(X) @ exp(Y)) in a numerically stable, where log/exp are entrywise operations.
#     X is assumed to be a sparse tensor, specified by indices and values (must be coalesced!).
#     The implementation relies on torch's built-in "logsumexp".
#     """
#     assert x_cols == y.shape[0]
#     assert x_values.dtype == y.dtype
#
#     print("start spmm helper!")
#     x_targets = [
#         torch.where(x_indices[0] == i)[0]
#         for i in range(x_rows)
#     ]
#
#     # z = torch.zeros((x_rows, y.shape[1]), device=x_values.device, dtype=x_values.dtype)
#     #
#     # pool = mp.Pool(cfg.model_cfg.num_cores)
#     # print("started pool")
#     #
#     # pool.starmap(_spmm_ij_handler, [
#     #     (x_indices, x_values, y, z, i, j)
#     #     for i, j in itertools.product(range(x_rows), range(y.shape[1]))
#     # ])
#
#     # futures = [
#     #     [
#     #         torch.jit.fork(_spmm_ij_handler, x_indices, x_values, y, i, j)
#     #         for j in range(y.shape[1])
#     #     ]
#     #     for i in range(x_rows)
#     # ]
#     # print("constructed futures")
#     #
#     # z = torch.tensor(
#     #     [
#     #         [
#     #             torch.jit.wait(future)
#     #             for future in future_row
#     #         ]
#     #         for future_row in futures
#     #     ],
#     #     device=x_values.device,
#     #     dtype=x_values.dtype
#     # )
#
#     z = torch.zeros((x_rows, y.shape[1]), device=x_values.device, dtype=x_values.dtype)
#     for i in range(x_rows):
#         futures = [torch.jit.fork(_spmm_ij_handler, x_indices, x_values, x_targets, y, i, j) for j in range(y.shape[1])]
#         for j in range(y.shape[1]):
#             z[i, j] = torch.jit.wait(futures[j])
#         print(f"Done with row {i}")
#
#     print("end spmm helper!")
#
#     return z


# @torch.jit.script
# def log_spspmm_exp_helper(x_indices: torch.Tensor,
#                         x_values: torch.Tensor,
#                         x_rows: int,
#                         x_cols: int,
#                         y_indices: torch.Tensor,
#                         y_values: torch.Tensor,
#                         y_rows: int,
#                         y_cols: int
#                         ) -> torch.Tensor:
#     """
#     Computes log(exp(X) @ exp(Y)) in a numerically stable, where log/exp are entrywise operations.
#     X is assumed to be a sparse tensor, specified by indices and values (must be coalesced!).
#     The implementation relies on torch's built-in "logsumexp".
#     """
#     assert x_cols == y_rows
#     assert x_values.dtype == y_values.dtype
#
#     z = torch.zeros((x_rows, y_cols), device=x_values.device, dtype=x_values.dtype)
#     for i in range(z.shape[0]):
#         for j in range(z.shape[1]):
#             x_targets, = torch.where(x_indices[0] == i)
#             y_targets, = torch.where(y_indices[1] == j)
#             _sum_of_logs = torch.zeros(x_cols, device=x_values.device, dtype=x_values.dtype)
#             _sum_of_logs[x_indices[1, x_targets]] += x_values[x_targets]
#             _sum_of_logs[y_indices[0, y_targets]] += y_values[y_targets]
#             z[i, j] = torch.logsumexp(_sum_of_logs, dim=0)
#     return z


# def log_spmm_exp(x: SparseMatrix, y: torch.Tensor):
#     """
#     Computes log(exp(X) @ exp(Y)) in a numerically stable, where log/exp are entrywise operations.
#     X is assumed to be a sparse tensor, specified by indices and values (must be coalesced!).
#     The implementation relies on torch's built-in "logsumexp".
#     """
#     return log_spmm_exp_helper(
#         x.indices,
#         x.values,
#         x.rows,
#         x.columns,
#         y
#     )
#
#
# def log_spspmm_exp(x: SparseMatrix, y: SparseMatrix):
#     return log_spspmm_exp_helper(
#         x.indices,
#         x.values,
#         x.rows,
#         x.columns,
#         y.indices,
#         y.values,
#         y.rows,
#         y.columns,
#     )
