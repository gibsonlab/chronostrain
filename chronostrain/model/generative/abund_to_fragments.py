from pathlib import Path
from typing import Tuple
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from chronostrain.util.math import save_sparse_matrix, load_sparse_matrix


class FragmentFrequencySparse(object):
    def __init__(self, matrix: jsparse.BCOO):
        self.matrix = matrix

    @property
    def num_strains(self) -> int:
        return self.matrix.shape[1]

    @property
    def num_frags(self) -> int:
        return self.matrix.shape[0]

    def save(self, path: Path):
        save_sparse_matrix(path, self.matrix)

    @staticmethod
    def load(path: Path) -> 'FragmentFrequencySparse':
        return FragmentFrequencySparse(load_sparse_matrix(path))

    def slice_by_fragment(self, frag_idx: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        locs, = jnp.where(self.matrix.indices[:, 0] == frag_idx)
        return self.matrix.indices[locs, 1], self.matrix.data[locs]

    def get_value(self, frag_idx: int, strain_idx: int) -> float:
        locs, = jnp.where(
            (self.matrix.indices == (frag_idx, strain_idx)).all(axis=1)
        )
        if locs.shape[0] > 1:
            raise ValueError(
                "Fragment frequency matrix had more than one hit for pair (f={}, s={})".format(
                    frag_idx, strain_idx
                )
            )
        elif locs.shape[0] == 0:
            raise ValueError(
                "Fragment frequency matrix had no hits for (f={}, s={})".format(
                    frag_idx, strain_idx
                )
            )
        loc = locs[0]
        return self.matrix.data[loc].item()
