from typing import Tuple, List

import torch
from .sparse_tensor import SparseMatrix


class ColumnSectionedSparseMatrix(SparseMatrix):
    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 dims: Tuple[int, int],
                 force_coalesce: bool = True):
        super().__init__(indices, values, dims, force_coalesce=force_coalesce)
        self.locs_per_column: List[torch.Tensor] = [
            torch.where(indices[1] == i)[0]
            for i in range(self.columns)
        ]

    def get_slice(self, col: int) -> torch.Tensor:
        return self.locs_per_column[col]

    @staticmethod
    def from_sparse_matrix(x: SparseMatrix):
        return ColumnSectionedSparseMatrix(
            x.indices,
            x.values,
            (x.rows, x.columns),
            False
        )


class RowSectionedSparseMatrix(SparseMatrix):
    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 dims: Tuple[int, int],
                 force_coalesce: bool = True):
        super().__init__(indices, values, dims, force_coalesce=force_coalesce)
        self.locs_per_row: List[torch.Tensor] = [
            torch.where(indices[0] == i)[0]
            for i in range(self.rows)
        ]

    def get_slice(self, row: int) -> torch.Tensor:
        return self.locs_per_row[row]
