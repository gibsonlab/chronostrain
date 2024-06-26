"""
A sparse 2-d matrix in COO format. Uses torch_sparse as a backend.

Note: since torch_sparse largely already uses C-compiled computations or JIT whenever possible, calls using
torch_sparse need not be optimized. For custom model-specific operations, we use @torch.jit ourselves.

TODO: remove dependency on torch_sparse.
"""
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
import torch
import torch_sparse


def size_of_tensor(x: torch.Tensor):
    return x.element_size() * x.nelement()


class SparseMatrix(object):
    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 dims: Union[List[int], Tuple[int, int]],
                 force_coalesce: bool = False):
        self.rows: int = int(dims[0])
        self.columns: int = int(dims[1])

        if force_coalesce:
            ind, val = torch_sparse.coalesce(indices, values, dims[0], dims[1])
        else:
            ind = indices
            val = values

        self.indices: torch.Tensor = ind
        self.values: torch.Tensor = val

    def size(self) -> torch.Size:
        return torch.Size((self.rows, self.columns))

    def physical_size(self) -> int:
        return size_of_tensor(self.indices) + size_of_tensor(self.values)

    def get(self, r: int, c: int):
        matches = (self.indices[0, :] == r) and (self.indices[1, :] == c)
        return torch.sum(self.values[matches])

    @property
    def nnz(self) -> int:
        return self.values.size()[0]

    def sparsity(self) -> float:
        """
        Measures the sparsity of the matrix, computed as (# of empty entries) / (row * cols).
        Note: this is the complement of x.density(), so that x.sparsity() + x.density() = 1.
        """
        return 1 - self.density()

    def density(self) -> float:
        """
        Measures the density of the matrix, computed as (# of nonempty entries) / (row * cols).
        Note: this is the complement of x.sparsity(), so that x.sparsity() + x.density() = 1.
        """
        if self.rows > 0 and self.columns > 0:
            return self.nnz / (self.rows * self.columns)
        else:
            return float("inf")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rows, self.columns

    def dense_mul(self, x: torch.Tensor) -> torch.Tensor:
        return torch_sparse.spmm(self.indices, self.values, self.rows, self.columns, x)

    def sparse_mul(self, x: 'SparseMatrix') -> 'SparseMatrix':
        if self.columns != x.rows:
            raise RuntimeError(f"Cannot multiply ({self.rows}x{self.columns}) and ({x.rows}x{x.columns}) matrices.")

        if self.rows == 0 or self.columns == 0:
            return SparseMatrix(
                torch.tensor([[], []], device=self.values.device, dtype=torch.long),
                torch.tensor([], device=self.values.device, dtype=self.values.dtype),
                (self.rows, x.columns),
                force_coalesce=False
            )

        result_indices, result_values = torch_sparse.spspmm(
            self.indices, self.values, x.indices, x.values, self.rows, self.columns, x.columns
        )
        return SparseMatrix(result_indices, result_values, (self.rows, x.columns), force_coalesce=False)

    def exp(self) -> 'SparseMatrix':
        """
        Component-wise exponentiation, applied to only the non-empty entries.
        Note: Be careful of what the empty values mean. If the empty values before the call to exp() are meant to
        be zeroes, then this is not consistent with the behavior in this implementation, since exp(0) = 1.
        """
        return SparseMatrix(self.indices, torch.exp(self.values), (self.rows, self.columns), force_coalesce=False)

    def log(self) -> 'SparseMatrix':
        """
        Component-wise logarithm, applie to only the non-empty entries.
        Note: Be careful of what the empty values mean. If the empty values before the call to log() are meant to
        be zeroes, then this is not consistent with the behavior in this implementation, since log(0) = -inf.
        """
        return SparseMatrix(self.indices, torch.log(self.values), (self.rows, self.columns), force_coalesce=False)

    def add(self, y: 'SparseMatrix') -> 'SparseMatrix':
        """
        :param y: The other matrix in the summand.
        :return: A SparseMatrix instance representing x + y, where x is this matrix.
        """
        if self.rows != y.rows:
            raise ValueError("The number of rows must match in both matrices.")
        if self.columns != y.columns:
            raise ValueError("The number of columns must match in both matrices.")

        return SparseMatrix(
            indices=torch.cat([self.indices, y.indices], dim=-1),
            values=torch.cat([self.values, y.values], dim=-1),
            dims=(self.rows, self.columns),
            force_coalesce=False
        )

    def add_constant(self, x: Union[int, float], inplace: bool = True) -> 'SparseMatrix':
        if inplace:
            self.values = self.values + x
            return self
        else:
            return SparseMatrix(
                self.indices,
                self.values + x,
                (self.rows, self.columns),
                force_coalesce=False
            )

    def t(self) -> 'SparseMatrix':
        return SparseMatrix(
            torch.stack([self.indices[1], self.indices[0]]),
            self.values,
            (self.columns, self.rows),
            force_coalesce=False
        )

    def to_dense(self, zero_fill=0.) -> torch.Tensor:
        if zero_fill == 0.:
            return torch.sparse_coo_tensor(
                self.indices,
                self.values,
                (self.rows, self.columns)
            ).to_dense()
        else:
            x = torch.ones((self.rows, self.columns), dtype=self.values.dtype, device=self.values.device) * zero_fill
            x[self.indices[0], self.indices[1]] = self.values
            return x

    def scale_row(self, vec: torch.Tensor, dim: int = 0) -> 'SparseMatrix':
        """
        Performs a row-scaling operation. Using dense matrices, this is equivalent to x * vec.unsqueeze(0).
        If dim = 0, interprets "vec" as a column vector so that the i-th row is scaled by the i-th entry.
        If dim = 1, interprets "vec" as a row vector so that the i-th column is scaled by the i-th entry.

        :param vec: The vector to scale by. The dimension must match the specified matrix dimension's size.
        :param dim: The dimension to scale.
        :return: A new SparseMatrix instance representing the output.
        """
        if len(vec.size()) > 1:
            raise ValueError("Can only row_scale by 1-d vectors.")
        if vec.size()[0] != [self.rows, self.columns][dim]:
            raise ValueError("The size of the input scale vector and the size of this "
                             "matrix (along the input dim) must match.")
        return SparseMatrix(
            self.indices,
            self.values * vec[self.indices[dim, :]],
            (self.rows, self.columns),
            force_coalesce=False
        )

    def sum(self, dim: int):
        """
        Sums the matrix along a specified input dimension.

        :param dim: The dimension to collapse.
        :return: A dense 1-d vector.
        """
        if self.nnz == 0:
            return torch.zeros(
                self.rows if dim == 1 else self.columns,
                dtype=self.values.dtype,
                device=self.values.device
            )
        x = torch.sparse_coo_tensor(self.indices, self.values, (self.rows, self.columns))
        return torch.sparse.sum(x, dim=dim).to_dense()

    def normalize(self, dim: int) -> 'SparseMatrix':
        """
        Normalizes the specified dimension of a 2-d matrix (so that summing along that dimension returns a tensor of ones).
        The input matrix is assumed to be sparse in a single dimension only (either row-sparse or column-sparse)
        stored as a sparse_coo tensor.

        Note: only efficient if `dim` is over a sufficiently sparse dimension. (The 1-d row/col sum matrix will be
        converted to dense.)
        """
        sums = self.sum(dim=dim)
        rescaled_values = self.values / sums[self.indices[1 - dim]]

        return SparseMatrix(self.indices, rescaled_values, (self.rows, self.columns), force_coalesce=False)

    def column_normed_row_sum(self) -> torch.Tensor:
        """
        Normalize each column (so that each column sums to 1 (sum, dim=0)), and then sum over the rows (sum, dim=1).

        :return: A dense 1-d vector.
        """
        x = torch.sparse_coo_tensor(self.indices, self.values, (self.rows, self.columns)).coalesce()
        sums = torch.sparse.sum(x, dim=0).to_dense()
        rescaled_values = x.values() / sums[x.indices()[1]]
        x = torch.sparse_coo_tensor(self.indices, rescaled_values, (self.rows, self.columns))
        return torch.sparse.sum(x, dim=1).to_dense()

    def sparse_slice(self, dim: int, idx: int):
        """
        Returns the dimension `dim`, index `idx` slice of the sparse matrix, as a sparse Nx1 matrix (column vector).
        Equivalent to x[idx, :].unsqueeze(1) if dim == 0, or x[:, idx].unsqueeze(1) if dim == 1.

        :param dim: The dimension to index into.
        :param idx: The index of the row/column to choose (depending on the specified dimension).
        :return: a column vector (N x 1).
        """
        matching_entries = torch.where(self.indices[dim, :] == idx)[0]
        collapsed_indices = self.indices[1 - dim, matching_entries]
        return SparseMatrix(
            indices=torch.stack([
                collapsed_indices,
                torch.zeros(size=collapsed_indices.size(), dtype=torch.long, device=collapsed_indices.device)
            ]),
            values=self.values[matching_entries],
            dims=(self.size()[1 - dim], 1),
            force_coalesce=False
        )

    def slice_columns(self, cols_to_keep: List[int]) -> 'SparseMatrix':
        """
        A column-slicing operation, equivalent to x[:, cols_to_keep].

        :param cols_to_keep:
        :return:
        """
        # Deleting entries
        cols_to_keep = torch.tensor(cols_to_keep, device=self.indices.device).unsqueeze(1)
        mask = torch.sum(torch.eq(cols_to_keep, self.indices[1, :]), dim=0).bool()
        indices = self.indices[:, mask]
        values = self.values[mask]

        print("[WARNING] TODO: SparseMatrix's implementation of `slice_columns` needs to be verified.")
        # check that this is correct.
        range_mask = torch.sum(torch.arange(self.columns, device=self.indices.device) == cols_to_keep, dim=0).bool()

        adj = torch.cumsum(~range_mask, 0)
        indices[1, :] = indices[1, :] - torch.index_select(adj, 0, indices[1, :])

        return SparseMatrix(indices, values, (self.rows, cols_to_keep.size(-2)), force_coalesce=False)

    def save(self, out_path: Path):
        np.savez(
            out_path,
            sparse_indices=self.indices.cpu().numpy(),
            sparse_values=self.values.cpu().numpy(),
            matrix_shape=np.array([
                self.rows,
                self.columns
            ])
        )

    @staticmethod
    def load(in_path: Path, device, dtype):
        data = np.load(str(in_path))
        size = data["matrix_shape"]
        return SparseMatrix(
            indices=torch.tensor(
                data['sparse_indices'],
                device=device,
                dtype=torch.long
            ),
            values=torch.tensor(
                data['sparse_values'],
                device=device,
                dtype=dtype
            ),
            dims=(size[0], size[1])
        )
