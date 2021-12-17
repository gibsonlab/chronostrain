from pathlib import Path
from typing import Tuple, List

import numpy as np
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
    def from_sparse_matrix(x: SparseMatrix) -> 'ColumnSectionedSparseMatrix':
        return ColumnSectionedSparseMatrix(
            x.indices,
            x.values,
            (x.rows, x.columns),
            True
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
    def from_sparse_matrix(x: SparseMatrix) -> 'RowSectionedSparseMatrix':
        return RowSectionedSparseMatrix(
            x.indices,
            x.values,
            (x.rows, x.columns),
            True
        )

    @staticmethod
    def load(in_path: Path, device, dtype):
        data = np.load(str(in_path))
        size = data["matrix_shape"]
        return RowSectionedSparseMatrix(
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
