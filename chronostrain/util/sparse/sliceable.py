import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from .sparse_tensor import SparseMatrix
import torch_scatter


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
            indices=x.indices,
            values=x.values,
            dims=(x.rows, x.columns),
            force_coalesce=False
        )


class RowSectionedSparseMatrix(SparseMatrix):
    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 dims: Tuple[int, int],
                 force_coalesce: bool = False,
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
            False
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


class BBVIOptimizedSparseMatrix(RowSectionedSparseMatrix):
    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 dims: Tuple[int, int],
                 row_chunk_size: int,
                 ):
        # Optimize by sorting by columns. (prepare for torch_scatter.segment_coo)
        sort_order = torch.argsort(indices[1])
        sorted_indices = indices[:, sort_order]
        sorted_values = values[sort_order]

        super().__init__(
            indices=sorted_indices,
            values=sorted_values,
            dims=dims,
            force_coalesce=False
        )

        # Retrieve the start/end locations of columns (assumed to be in sorted order).
        num_occurrences = torch.tensor([
            sorted_indices[1].eq(i).sum()
            for i in range(dims[1])
        ])
        self.col_indptrs = torch.concat([
            torch.zeros(1, dtype=num_occurrences.dtype, device=num_occurrences.device),
            torch.cumsum(num_occurrences, dim=0)
        ])

        # Precalculate all chunk locations.
        self.row_chunk_size = row_chunk_size

        num_chunks = math.ceil(self.rows / self.row_chunk_size)
        self.chunks: List[RowSectionedSparseMatrix] = []
        self.chunk_locs: List[torch.Tensor] = []

        for chunk_idx in range(num_chunks):
            chunk_start_idx = row_chunk_size * chunk_idx
            chunk_end_idx = min(chunk_start_idx + row_chunk_size, self.rows)

            chunk_loc = torch.where((chunk_start_idx <= self.indices[0]) & (self.indices[0] < chunk_end_idx))[0]
            self.chunk_locs.append(chunk_loc)

            self.chunks.append(RowSectionedSparseMatrix(
                indices=torch.stack([
                    self.indices[0][chunk_loc] - chunk_start_idx,
                    self.indices[1][chunk_loc]
                ], dim=0),
                values=self.values[chunk_loc],
                dims=(chunk_end_idx - chunk_start_idx, self.columns),
                force_coalesce=False
            ))

    def columnwise_min(self) -> torch.Tensor:
        return self.scatter_groupby_reduce('min')

    def columnwise_max(self) -> torch.Tensor:
        return self.scatter_groupby_reduce('max')

    def scatter_groupby_reduce(self, operation: str) -> torch.Tensor:
        ans = torch.empty(self.columns, device=self.values.device, dtype=self.values.dtype)
        torch_scatter.segment_csr(
            src=self.values,
            indptr=self.col_indptrs,
            reduce=operation,
            out=ans
        )
        return ans

    def update_values(self, new_values: torch.Tensor):
        self.values = new_values
        for chunk, chunk_loc in zip(self.chunks, self.chunk_locs):
            chunk.values = self.values[chunk_loc]

    @staticmethod
    def optimize_from_sparse_matrix(x: SparseMatrix, row_chunk_size: int) -> 'BBVIOptimizedSparseMatrix':
        return BBVIOptimizedSparseMatrix(
            indices=x.indices,
            values=x.values,
            dims=(x.rows, x.columns),
            row_chunk_size=row_chunk_size
        )

    def copy_pattern(self) -> 'BBVIOptimizedSparseMatrix':
        return BBVIOptimizedSparseMatrix(
            indices=self.indices,
            values=torch.empty(self.values.shape, device=self.values.device, dtype=self.values.dtype),
            dims=(self.rows, self.columns),
            row_chunk_size=self.row_chunk_size,
        )

    def collect_chunk(self, chunk_idx: int, chunk: RowSectionedSparseMatrix):
        chunk_start_idx = self.row_chunk_size * chunk_idx

        for chunk_row, chunk_row_locs in enumerate(chunk.locs_per_row):
            self.values[self.locs_per_row[chunk_row + chunk_start_idx]] = chunk.values[chunk_row_locs]

        self.chunks[chunk_idx] = chunk
