from abc import abstractmethod, ABC, ABCMeta
from pathlib import Path
from typing import Optional

import jax.numpy as np
from .constants import GENERIC_SAMPLE_TYPE, GENERIC_PARAM_TYPE


class AbstractPosterior(metaclass=ABCMeta):
    @abstractmethod
    def abundance_sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Returns a sample from this posterior distribution.
        :param num_samples: the number of samples (N).
        :return: A time-indexed, simplex-valued (T x N x S) abundance tensor.
        """
        pass

    @abstractmethod
    def log_likelihood(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def save(self, target_path: Path):
        pass


class AbstractReparametrizedPosterior(AbstractPosterior, ABC):
    def __init__(self, params: Optional[GENERIC_PARAM_TYPE] = None):
        if params is None:
            self.parameters = self.initial_params()
        else:
            self.parameters = params

    @abstractmethod
    def initial_params(self) -> GENERIC_PARAM_TYPE:
        raise NotImplementedError()

    def log_likelihood(self, samples: np.ndarray, params: GENERIC_PARAM_TYPE = None) -> np.ndarray:
        pass

    @abstractmethod
    def entropy(self, params: GENERIC_PARAM_TYPE, *args) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def random_sample(self, num_samples: int) -> GENERIC_SAMPLE_TYPE:
        """
        Return randomized samples (before reparametrization.)
        :param num_samples:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def get_parameters(self) -> GENERIC_PARAM_TYPE:
        raise NotImplementedError()

    @abstractmethod
    def set_parameters(self, params: GENERIC_PARAM_TYPE):
        """
        Store the value of these params internally as the state of this posterior.
        :param params: A list of parameter arrays (the implementation should decide the ordering.)
        :return:
        """
        pass

    def reparametrize(self, random_samples: GENERIC_SAMPLE_TYPE, params: GENERIC_PARAM_TYPE, *args) -> GENERIC_SAMPLE_TYPE:
        raise NotImplementedError()

    def save(self, path: Path):
        np.savez(
            str(path),
            **self.parameters
        )

    def load(self, path: Path):
        f = np.load(str(path))
        self.parameters = dict(f)
