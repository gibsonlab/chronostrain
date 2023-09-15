from typing import Iterator

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as cnp


def divide_columns_into_batches_sparse(x: jsparse.BCOO, batch_size: int) -> Iterator[jsparse.BCOO]:
    r, c = x.indices[:, 0], x.indices[:, 1]
    x_cols = x.shape[1]
    for i in range(0, x_cols, batch_size):
        start = i
        end = min(x_cols, i + batch_size)
        sz = end - start
        locs, = cnp.where((c >= start) & (c < end))

        if len(locs) == 0:
            continue

        yield jsparse.BCOO(
            (
                x.data[locs],
                x.indices[locs, :] - jnp.array([0, start], dtype=x.indices.dtype)
            ),
            shape=(x.shape[0], sz)
        )


def divide_columns_into_batches(x: jnp.ndarray, batch_size: int) -> Iterator[jnp.ndarray]:
    permutation = cnp.randperm(x.shape[1])
    for i in range(0, x.shape[1], batch_size):
        indices = permutation[i:i+batch_size]
        yield x[:, indices]


def log_dot_exp(x: jnp.ndarray, y: jnp.ndarray):  #([a], [a]) -> []
    return jax.scipy.special.logsumexp(x + y)


log_mv_exp = jax.vmap(log_dot_exp, (0, None), 0)  # ([b,a], [a]) -> [b]
log_mm_exp = jax.vmap(log_mv_exp, (None, 1), 1)  # ([b,a], [a, c]) -> [b, c]
