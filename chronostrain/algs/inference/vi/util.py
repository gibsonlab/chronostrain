from typing import Iterator

import jax.numpy as jnp
import numpy as cnp
import jax.experimental.sparse as jsparse


def divide_columns_into_batches_sparse(x: jsparse.BCOO, batch_size: int) -> Iterator[jsparse.BCOO]:
    r, c = x.indices[:, 0], x.indices[:, 1]
    for i in range(0, x.columns, batch_size):
        start = i
        end = min(x.columns, i + batch_size)
        sz = end - start
        locs, = jnp.where((c >= start) & (c < end))
        yield jsparse.BCOO(
            (x.values[locs], x.indices[locs, :]),
            shape=(x.rows, sz)
        )


def divide_columns_into_batches(x: jnp.ndarray, batch_size: int) -> Iterator[jnp.ndarray]:
    permutation = cnp.randperm(x.shape[1])
    for i in range(0, x.shape[1], batch_size):
        indices = permutation[i:i+batch_size]
        yield x[:, indices]
