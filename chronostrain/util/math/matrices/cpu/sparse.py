import math
import torch
import numpy as np
from numba import njit
from numba.typed import List as nList

from ..sliceable import ColumnSectionedSparseMatrix, RowSectionedSparseMatrix


@njit
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
    log_spspmm_exp_helper(
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
