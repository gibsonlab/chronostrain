from typing import TypeVar, Generic, Dict, Tuple
from pathlib import Path
import pickle

from chronostrain.model import SequenceRead


T = TypeVar('T')
K = TypeVar('K')


class UniqueHitOrdering(Generic[T, K]):
    def __init__(self):
        self.seen: Dict[K, int] = dict()

    def extract_key(self, x: T) -> K:
        raise NotImplementedError()

    def get_index_of(self, x: T) -> int:
        _key: K = self.extract_key(x)
        return self.get_index_from_key(_key)

    def get_index_from_key(self, k: K) -> int:
        if k in self.seen:
            return self.seen[k]
        else:
            next_idx = len(self.seen)
            self.seen[k] = next_idx
            return next_idx

    def save(self, p: Path):
        with open(p, 'wb') as f:
            pickle.dump(self.seen, f)

    @staticmethod
    def load(p: Path):
        with open(p, 'rb') as f:
            return pickle.load(f)

    def __len__(self) -> int:
        return len(self.seen)


class UniqueReadOrdering(UniqueHitOrdering[SequenceRead, str]):
    def extract_key(self, x: SequenceRead) -> str:
        return x.id


class UniquePairedReadOrdering(UniqueHitOrdering[Tuple[SequenceRead, SequenceRead], Tuple[str, str]]):
    def extract_key(self, x: Tuple[SequenceRead, SequenceRead]) -> Tuple[str, str]:
        return x[0].id, x[1].id
