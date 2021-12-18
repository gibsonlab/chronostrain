import math
from pathlib import Path
from typing import Tuple, List, Iterator

import numpy as np
import torch
from .sparse_tensor import SparseMatrix


class ColumnSectionedSparseMatrix(SparseMatrix):
    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 dims: Tuple[int, int],
                 force_coalesce: bool = True,
                 _explicit_locs_per_col: List[torch.Tensor] = None):
        super().__init__(indices, values, dims, force_coalesce=force_coalesce)
        if _explicit_locs_per_col is None:
            self.locs_per_column: List[torch.Tensor] = [
                torch.where(indices[1] == i)[0]
                for i in range(self.columns)
            ]
        else:
            self.locs_per_column = _explicit_locs_per_col

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
                 force_coalesce: bool = True,
                 _explicit_locs_per_row: List[torch.Tensor] = None):
        super().__init__(indices, values, dims, force_coalesce=force_coalesce)
        if _explicit_locs_per_row is None:
            self.locs_per_row: List[torch.Tensor] = [
                torch.where(indices[0] == i)[0]
                for i in range(self.rows)
            ]
        else:
            self.locs_per_row = _explicit_locs_per_row

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

    def transpose(self) -> ColumnSectionedSparseMatrix:
        return ColumnSectionedSparseMatrix(
            indices=torch.concat([self.indices[1], self.indices[0]]),
            values=self.values,
            dims=(self.columns, self.rows),
            force_coalesce=False,
            _explicit_locs_per_col=self.locs_per_row
        )

    def divide_into_chunks(self, chunk_size: int) -> Iterator['RowSectionedSparseMatrix']:
        num_chunks = math.ceil(self.rows / chunk_size)

        for chunk_idx in range(num_chunks):
            chunk_start_idx = chunk_size * chunk_idx
            chunk_end_idx = min(chunk_start_idx + chunk_size, self.rows)

            chunk_locs = torch.where((chunk_start_idx <= self.indices[0]) & (self.indices[0] < chunk_end_idx))[0]

            yield RowSectionedSparseMatrix(
                indices=torch.stack([
                    self.indices[0][chunk_locs] - chunk_start_idx,
                    self.indices[1][chunk_locs]
                ], dim=0),
                values=self.values[chunk_locs],
                dims=(chunk_end_idx - chunk_start_idx, self.columns),
                force_coalesce=False,
                _explicit_locs_per_row=self.locs_per_row[chunk_start_idx:chunk_end_idx]
            )

    @staticmethod
    def concatenate_chunks(chunks: List['RowSectionedSparseMatrix']) -> 'RowSectionedSparseMatrix':
        chunk_indices = []
        result_rows = 0
        result_columns = chunks[0].columns
        locs_per_row = []

        for chunk in chunks:
            assert chunk.columns == result_columns

            # Convert relative chunk rows to absolute rows
            absolute_chunk_indices = torch.clone(chunk.indices)
            absolute_chunk_indices[0] += result_rows
            chunk_indices.append(absolute_chunk_indices)

            # Join the row sparsity indexing.
            locs_per_row += chunk.locs_per_row

            # Increment the number of rows seen so far.
            result_rows += chunk.rows

        return RowSectionedSparseMatrix(
            indices=torch.concat(chunk_indices, dim=1),
            values=torch.concat([chunk.values for chunk in chunks]),
            dims=(result_rows, result_columns),
            force_coalesce=False,
            _explicit_locs_per_row=locs_per_row
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
