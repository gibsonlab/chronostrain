from typing import Tuple, Iterator, Dict
import pickle
from pathlib import Path
from .fragment import Fragment


class FragmentPairNotFound(BaseException):
    def __init__(self, f1: Fragment, f2: Fragment):
        self.frag1 = f1
        self.frag2 = f2


class FragmentPairSpace(object):
    """
    Models ordered pairs of the form (f1, f2), where f1 and f2 are fragment objects.
    """
    def __init__(self, precomputed_index: Dict[Tuple[int, int], int] = None):
        if precomputed_index is None:
            self.seen_pairs: Dict[Tuple[int, int], int] = {}
        else:
            self.seen_pairs = precomputed_index

    def get_index(self, frag1: Fragment, frag2: Fragment, insert: bool = True) -> int:
        k = (frag1.index, frag2.index)
        if k in self.seen_pairs:
            return self.seen_pairs[k]
        else:
            if insert:
                new_idx = len(self.seen_pairs)
                self.seen_pairs[k] = new_idx
                return new_idx
            else:
                raise FragmentPairNotFound(frag1, frag2)

    def save(self, path: Path):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, 'wb') as f:
            pickle.dump(self.seen_pairs, f)

    @staticmethod
    def load(path: Path) -> 'FragmentPairSpace':
        with open(path, 'rb') as f:
            return FragmentPairSpace(pickle.load(f))

    def __len__(self) -> int:
        return len(self.seen_pairs)

    def __iter__(self) -> Iterator[Tuple[int, int, int]]:
        for (i1, i2), i_pair in self.seen_pairs.items():
            yield i1, i2, i_pair
