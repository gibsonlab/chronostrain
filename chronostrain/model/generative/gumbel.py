from typing import List
import jax
import jax.numpy as np
import numpy as cnp
import jax.scipy as scipy
from scipy.special import log_expit as c_log_expit


class PopulationGlobalZeros(object):
    def __init__(self, num_strains: int, prior_p: float=0.5):
        self.num_strains = num_strains
        self.prior_p = prior_p
        self.ONE_logp = cnp.log(prior_p)
        self.ZERO_logp = cnp.log(1 - prior_p)
        self.log_denominator = c_log_expit(-self.num_strains * cnp.log(1 - prior_p))  # LOG[1 / (1 - p(all zeros))]

    def log_likelihood(self, booleans: np.ndarray) -> np.ndarray:
        """
        @param booleans: An (N x S) tensor of zeros or ones. (likelihood won't depend on smoothness)
        @return: a length-N tensor of likelihoods, one per sample.
        """
        # likelihood actually doesn't depend on the actual zeros/ones since prior is Bernoulli(0.5),
        # conditioned on not all being zero.
        return np.sum(
            self.ONE_logp * booleans + self.ZERO_logp * (1 - booleans),
            axis=-1
        ) + self.log_denominator


# ========================
# These are sometimes useful helper functions.
# ========================
def _expect_tensor_shape(x: np.ndarray, name: str, shape: List[int]):
    if (len(x.shape) != len(shape)) or (list(x.shape) != shape):
        raise ValueError("Tensor `{}` must be of size {}. Got: {}".format(
            name,
            shape,
            x.shape
        ))


def _smooth_argmax(logits: np.ndarray, inv_temp: float, axis: int) -> np.ndarray:
    return jax.nn.softmax(inv_temp * logits, axis=axis)


def _smooth_log_argmax(logits: np.ndarray, inv_temp: float, axis: int) -> np.ndarray:
    """
    Logits are assumed to be stacked along axisension 0.
    """
    return jax.nn.log_softmax(inv_temp * logits, axis=axis)


def _smooth_log_p(logits: np.ndarray, inv_temp: float) -> np.ndarray:
    return inv_temp * logits[0] - scipy.special.logsumexp(logits * inv_temp, axis=0)


def _gumbel_logpdf(x: np.ndarray):
    return -x - np.exp(-x)


def _smooth_max(x: np.ndarray, inv_temp: float, axis: int) -> np.ndarray:
    """
    x is assumed to be stacked along axisension 0.
    """
    # TODO: inspect this behavior. does softmax output nan as expected when multiplying 0 by -inf?
    return np.nansum(jax.nn.softmax(inv_temp * x, axis=axis) * x, axis=axis)
