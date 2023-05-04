from typing import Iterator

import torch
import torch.nn.functional

from chronostrain.util.math.matrices import SparseMatrix
from chronostrain.config import cfg


def divide_columns_into_batches_sparse(x: SparseMatrix, batch_size: int) -> Iterator[SparseMatrix]:
    r, c = x.indices[0], x.indices[1]
    for i in range(0, x.columns, batch_size):
        start = i
        end = min(x.columns, i + batch_size)
        sz = end - start
        locs, = torch.where((c >= start) & (c < end))
        yield SparseMatrix(
            indices=x.indices[:, locs],
            values=x.values[locs],
            dims=(x.rows, sz)
        )


def divide_columns_into_batches(x: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]:
    permutation = torch.randperm(x.shape[1], device=cfg.torch_cfg.device)
    for i in range(0, x.shape[1], batch_size):
        indices = permutation[i:i+batch_size]
        yield x[:, indices]
