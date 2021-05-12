from typing import Tuple, List
import torch
import torch_sparse


class CoalescedSparseMatrix(object):
    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 dims: Tuple[int, int],
                 force_coalesce: bool = True):
        self.rows: int = dims[0]
        self.columns: int = dims[1]

        if force_coalesce:
            ind, val = torch_sparse.coalesce(indices, values, dims[0], dims[1])
        else:
            ind = indices
            val = values

        self.indices: torch.Tensor = ind
        self.values: torch.Tensor = val

    def dense_mul(self, x: torch.Tensor) -> torch.Tensor:
        return torch_sparse.spmm(self.indices, self.values, self.rows, self.columns, x)

    def sparse_mul(self, x: 'CoalescedSparseMatrix') -> 'CoalescedSparseMatrix':
        result_indices, result_values = torch_sparse.spspmm(
            self.indices, self.values, x.indices, x.values, self.rows, self.columns, x.columns
        )
        return CoalescedSparseMatrix(result_indices, result_values, (self.rows, x.columns))

    def exp(self) -> 'CoalescedSparseMatrix':
        return CoalescedSparseMatrix(self.indices, torch.exp(self.values), (self.rows, self.columns))

    def t(self) -> 'CoalescedSparseMatrix':
        result_indices, result_values = torch_sparse.transpose(self.indices, self.values, self.rows, self.columns)
        return CoalescedSparseMatrix(result_indices, result_values, (self.columns, self.rows))

    def densify(self) -> torch.Tensor:
        intermediate = torch.sparse_coo_tensor(self.indices, self.values, (self.rows, self.columns))
        return intermediate.to_dense()

    def sparse_scalar_sum(self, scalar: float) -> 'CoalescedSparseMatrix':
        return CoalescedSparseMatrix(self.indices, self.values + scalar, (self.rows, self.columns))

    def row_hadamard_dense_vector(self, vec: torch.Tensor) -> 'CoalescedSparseMatrix':
        """
        DEPRECATED. Use "scale_row" instead, with `dim = 0` passed in. (double check this!)
        """
        dv = vec.repeat(1, self.columns)[self.indices[0, :], self.indices[1, :]]
        return CoalescedSparseMatrix(self.indices, self.values * dv, (self.rows, self.columns))

    def scale_row(self, vec: torch.Tensor, dim: int = 0) -> 'CoalescedSparseMatrix':
        if len(vec.size()) > 1:
            raise ValueError("Can only row_scale by 1-d vectors.")
        if vec.size()[0] != [self.rows, self.columns][dim]:
            raise ValueError("The size of the input scale vector and the size of this "
                             "matrix (along the input dim) must match.")
        return CoalescedSparseMatrix(
            self.indices,
            self.values * vec[self.indices[dim, :]],
            (self.rows, self.columns)
        )

    def sum(self, dim: int):
        x = torch.sparse_coo_tensor(self.indices, self.values, (self.rows, self.columns)).coalesce()
        return torch.sparse.sum(x, dim=dim).to_dense()

    def column_normed_row_sum(self) -> torch.Tensor:
        """
        Normalize each column (so that each column sums to zero (sum, dim=0)), and then sum over the rows (sum, dim=1)
        :return:
        """
        x = torch.sparse_coo_tensor(self.indices, self.values, (self.rows, self.columns)).coalesce()
        sums = torch.sparse.sum(x, dim=0).to_dense()
        rescaled_values = x.values() / sums[x.indices()[1]]
        x = torch.sparse_coo_tensor(self.indices, rescaled_values, (self.rows, self.columns))
        return torch.sparse.sum(x, dim=1).to_dense()

    def slice_cols(self, cols_to_keep: List[int]) -> 'CoalescedSparseMatrix':
        # Deleting entries
        cols_to_keep = torch.tensor(cols_to_keep).unsqueeze(1)
        mask = torch.sum(cols_to_keep == self.indices[1, :], dim=0).bool()
        indices = self.indices[:, mask]
        values = self.values[mask]

        # Shifting remaining entries down
        range_mask = torch.sum(torch.arange(self.columns) == cols_to_keep, dim=0).bool()
        adj = torch.cumsum(~range_mask, 0)
        indices[1, :] = indices[1, :] - torch.index_select(adj, 0, indices[1, :])

        return CoalescedSparseMatrix(indices, values, (self.rows, cols_to_keep.size(-2)))

