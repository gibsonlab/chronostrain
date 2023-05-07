from abc import ABC, abstractmethod
import jax.numpy as jnp


class FragmentLengthPrior(ABC):

    @abstractmethod
    def mean(self):
        raise NotImplementedError()

    @abstractmethod
    def std(self):
        raise NotImplementedError()

    @abstractmethod
    def var(self):
        raise NotImplementedError()

    @abstractmethod
    def logpmf(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()
