from typing import List

import torch
import torch_scatter

from chronostrain.util.sparse import SparseMatrix, RowSectionedSparseMatrix
from chronostrain.util.sparse.sliceable import ADVIOptimizedSparseMatrix
from chronostrain.config import cfg


def log_softmax(x_samples: torch.Tensor, t: int) -> torch.Tensor:
    # x_samples: (T x N x S) tensor.
    return x_samples[t] - torch.logsumexp(x_samples[t], dim=1, keepdim=True)


def log_spherical(x_samples: torch.Tensor, t: int, eps=1e-30) -> torch.Tensor:
    # x_samples: (T x N x S) tensor.
    square = torch.pow(x_samples[t], 2) + eps
    return torch.log(square) - torch.log(square.sum(dim=-1, keepdim=True))


def log_taylor(x_samples: torch.Tensor, t: int) -> torch.Tensor:
    exp_taylor = 1 + x_samples[t] + 0.5 * torch.pow(x_samples[t], 2)
    return torch.log(exp_taylor) - torch.log(exp_taylor.sum(dim=-1, keepdim=True))


def log_matmul_exp(x_samples: torch.Tensor, read_lls_batch: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(
        x_samples.unsqueeze(2) + read_lls_batch.unsqueeze(0),  # (N x S) @ (S x R_batch)
        dim=1,
        keepdim=False
    )


def divide_columns_into_batches(x: torch.Tensor, batch_size: int):
    permutation = torch.randperm(x.shape[1], device=cfg.torch_cfg.device)
    for i in range(0, x.shape[1], batch_size):
        indices = permutation[i:i+batch_size]
        yield x[:, indices]


class LogMMExpModel(torch.nn.Module):
    """
    Represents a Module which represents
        f_A(X) = log_matmul_exp(X, A)
    where A is a (D x E) dense matrix, and X is a (N x D) dense matrix.
    """
    def __init__(self, right_matrix: torch.Tensor):
        super().__init__()
        self.A: torch.Tensor = right_matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(
            x.unsqueeze(2) + self.A.unsqueeze(0),
            dim=1,
            keepdim=False
        )


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


class LogMMExpDenseSPModel_Async(torch.nn.Module):
    """
    Represents a Module which represents
        f_A(X) = log_matmul_exp(X, A)
    where A is a (D x E) SparseMatrix, and X is an (N x D) matrix.
    """
    def __init__(self, sparse_right_matrix: SparseMatrix, row_chunk_size: int = 100):
        super().__init__()
        if not isinstance(sparse_right_matrix, ADVIOptimizedSparseMatrix):
            self.A = ADVIOptimizedSparseMatrix.optimize_from_sparse_matrix(
                sparse_right_matrix,
                row_chunk_size=row_chunk_size
            )
        else:
            self.A = sparse_right_matrix

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
