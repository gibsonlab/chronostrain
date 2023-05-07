import jax
import jax.numpy as jnp
import numpy as np

from .priors import FragmentLengthPrior

class FragmentNegbinPrior(FragmentLengthPrior):
    def __init__(self, n: int, p: float):
        self.n = n
        self.p = p

    def mean(self):
        return self.n * (1 - self.p) / self.p

    def std(self):
        return np.sqrt(self.var)

    def var(self):
        return self.n * (1 - self.p) / np.square(self.p)

    def logpmf(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.scipy.stats.nbinom.logpmf(x, n=self.n, p=self.p)
