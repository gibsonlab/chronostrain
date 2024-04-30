from pathlib import Path
from jax.experimental.sparse import BCOO
from chronostrain.util.math import save_sparse_matrix, load_sparse_matrix


class FragmentReadErrorLikelihood(object):
    def __init__(self, matrix: BCOO):
        self.matrix: BCOO = matrix

    @property
    def num_reads(self) -> int:
        return self.matrix.shape[1]

    @property
    def num_frags(self) -> int:
        return self.matrix.shape[0]

    def save(self, path: Path):
        save_sparse_matrix(path, self.matrix)

    @staticmethod
    def load(path: Path) -> 'FragmentReadErrorLikelihood':
        return FragmentReadErrorLikelihood(load_sparse_matrix(path))
