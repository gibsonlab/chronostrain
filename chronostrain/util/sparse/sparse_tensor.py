import torch
import torch_sparse

class coalesced_sparse_tensor:
    def __init__(self, indices: torch.tensor, values: torch.tensor, dims: tuple):
        self.rows = dims[0]
        self.columns = dims[1]
        self.indices, self.values = torch_sparse.coalesce(indices, values, dims[0], dims[1])

    def dense_mul(self, x: torch.tensor) -> torch.tensor:
        return torch_sparse.spmm(self.indices, self.values, self.rows, self.columns, x)

    def sparse_mul(self, x: 'coalesced_sparse_tensor') -> 'coalesced_sparse_tensor':
        result_indices, result_values = torch_sparse.spspmm(
            self.indices, self.values, x.indices, x.values, self.rows, self.columns, x.columns
        )
        return coalesced_sparse_tensor(result_indices, result_values, (self.rows, x.columns))

    def exp(self) -> 'coalesced_sparse_tensor':
        return coalesced_sparse_tensor(self.indices, torch.exp(self.values), (self.rows, self.columns))

    def t(self) -> 'coalesced_sparse_tensor':
        result_indices, result_values = torch_sparse.transpose(self.indices, self.values, self.rows, self.columns)
        return coalesced_sparse_tensor(result_indices, result_values, (self.columns, self.rows))

    def densify(self) -> torch.tensor:
        intermediate = torch.sparse_coo_tensor(self.indices, self.values, (self.rows, self.columns))
        return intermediate.to_dense()

    def sparse_scalar_sum(self, scalar: float) -> 'coalesced_sparse_tensor':
        return coalesced_sparse_tensor(self.indices, self.values + scalar, (self.rows, self.columns))

    def row_hadamard(self, vec) -> 'coalesced_sparse_tensor':
        dv = vec.repeat(1,self.columns)[self.indices[0,:], self.indices[1,:]]
        return coalesced_sparse_tensor(self.indices, self.values*dv, (self.rows, self.columns))

    def sum(self, dim):
        x = torch.sparse_coo_tensor(self.indices, self.values, (self.rows, self.columns)).coalesce()
        return torch.sparse.sum(x, dim=0).to_dense()

    def column_normed_row_sum(self):
        x = torch.sparse_coo_tensor(self.indices, self.values, (self.rows, self.columns)).coalesce()
        sums = torch.sparse.sum(x, dim=0).to_dense()
        rescaled_values = x.values() / sums[x.indices()[1]]
        x = torch.sparse_coo_tensor(self.indices, rescaled_values, (self.rows, self.columns))
        return torch.sparse.sum(x, dim=1).to_dense()

    def del_cols(self, cols_to_keep: torch.tensor) -> 'coalesced_sparse_tensor':
        # Deleting entries
        cols_to_keep = cols_to_keep.t()
        mask = (self.indices[1,:] == cols_to_keep).sum(0).bool()
        indices = self.indices[:,mask]
        values = self.values[mask]

        # Shifting remaining entries down
        range_mask = (torch.arange(self.columns) == cols_to_keep).sum(0).bool()
        adj = torch.cumsum(~range_mask, 0)
        indices[1,:] = indices[1,:] - torch.index_select(adj, 0, indices[1,:])

        return coalesced_sparse_tensor(indices, values, (self.rows, cols_to_keep.size(-2)))

