from typing import List

import torch
from chronostrain.util.sparse import SparseMatrix, RowSectionedSparseMatrix
from torch_scatter import scatter


def log_softmax(x_samples: torch.Tensor, t: int) -> torch.Tensor:
    # x_samples: (T x N x S) tensor.
    return x_samples[t] - torch.logsumexp(x_samples[t], dim=1, keepdim=True)


class LogMMExpDenseSPModel(torch.nn.Module):
    """
    Represents a Module which represents
        f_A(X) = log_matmul_exp(X, A)
    where A is a (D x E) SparseMatrix, and X is an (N x D) matrix.
    """
    def __init__(self, sparse_right_matrix: SparseMatrix):
        super().__init__()
        self.A_indices: torch.Tensor = sparse_right_matrix.indices
        self.A_values: torch.Tensor = sparse_right_matrix.values
        self.A_rows: int = int(sparse_right_matrix.rows)
        self.A_columns: int = int(sparse_right_matrix.columns)

        self.nz_targets: List[torch.Tensor] = []
        self.target_cols: List[torch.Tensor] = []

        for target_row in range(self.A_rows):
            if isinstance(sparse_right_matrix, RowSectionedSparseMatrix):
                nz_targets_k = sparse_right_matrix.locs_per_row[target_row]
            else:
                nz_targets_k = torch.where(sparse_right_matrix.indices[0] == target_row)[0]
            self.nz_targets.append(nz_targets_k)
            self.target_cols.append(self.A_indices[1, nz_targets_k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = torch.full(
            fill_value=-float('inf'),
            size=[x.shape[0], self.A_columns],
            device=self.A_values.device,
            dtype=self.A_values.dtype
        )

        for target_idx in range(self.A_rows):
            """
            Given a target index k, compute the k-th summand of the dot product <u,v> = SUM_k u_k v_k,
            for each row u of x, and each column v of y.

            Note that k here specifies a column of x, and a row of y.
            """
            nz_targets: torch.Tensor = self.nz_targets[target_idx]
            target_cols: torch.Tensor = self.target_cols[target_idx]

            next_sum: torch.Tensor = x[:, target_idx].view(-1, 1) + self.A_values[nz_targets].view(1, -1)
            result[:, target_cols] = torch.logsumexp(
                torch.stack([result[:, target_cols], next_sum], dim=0),
                dim=0
            )

        return result


from chronostrain.util.sparse.sliceable import BBVIOptimizedSparseMatrix
import torch_scatter

class LogMMExpDenseSPModel_Async(torch.nn.Module):
    """
    Represents a Module which represents
        f_A(X) = log_matmul_exp(X, A)
    where A is a (D x E) SparseMatrix, and X is an (N x D) matrix.
    """
    def __init__(self, sparse_right_matrix: SparseMatrix):
        super().__init__()
        self.A = BBVIOptimizedSparseMatrix.optimize_from_sparse_matrix(
            sparse_right_matrix,
            row_chunk_size=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expansion = self.A.values.unsqueeze(0) + x[:, self.A.indices[0]]
        maximums = torch_scatter.segment_csr(
            src=expansion,
            indptr=self.A.col_indptrs.expand(x.shape[0], -1),
            reduce='max'
        )

        sumexp_offset = torch_scatter.segment_csr(
            src=torch.exp(expansion - maximums[:, self.A.indices[1]]),
            indptr=self.A.col_indptrs.expand(x.shape[0], -1),
            reduce='sum'
        )
        return maximums + sumexp_offset.log()


if __name__ == "__main__":
    A = SparseMatrix(
        indices=torch.tensor([
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ], dtype=torch.long),
        values=torch.tensor([1, 2, 3, 4], dtype=torch.float),
        dims=(2, 2)
    )

    B = torch.tensor([[3, 4], [0.5, 1]], dtype=torch.float)

    print(A.to_dense())
    m = LogMMExpDenseSPModel_Async(sparse_right_matrix=A.log().add_constant(-100, inplace=True))

    print("TEST ANSWER")
    print(m.forward(B.log() - 100))

    print("TRUE ANSWER")
    print(torch.log(B @ A.to_dense()) - 200)