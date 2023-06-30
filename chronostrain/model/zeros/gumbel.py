from typing import List
import jax
import jax.numpy as np
import numpy as cnp
import jax.scipy as scipy
from scipy.special import log_expit as c_log_expit


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


# class PopulationLocalZeros(object):
#     def __init__(self, time_points: List[float], num_strains: int):
#         self.time_points = time_points
#         self.num_strains = num_strains
#
#     def _validate_shapes(self, main_nodes: np.ndarray, between_nodes: np.ndarray):
#         n_samples = main_nodes.shape[2]
#         _expect_tensor_shape(main_nodes, "main_nodes", [2, len(self.time_points), n_samples, self.num_strains])
#         _expect_tensor_shape(between_nodes, "between_nodes", [2, len(self.time_points) - 1, n_samples, self.num_strains])
#
#     def log_likelihood(self, main_nodes: np.ndarray, between_nodes: np.ndarray) -> np.ndarray:
#         # both tensors are (2 x T x N x S).
#         self._validate_shapes(main_nodes, between_nodes)
#         return _gumbel_logpdf(
#             main_nodes
#         ).sum(axis=0).sum(axis=0).sum(axis=-1) + _gumbel_logpdf(
#             between_nodes
#         ).sum(axis=0).sum(axis=0).sum(axis=-1)
#
#     def zeroes_of_gumbels(self, main_nodes: np.ndarray, between_nodes: np.ndarray) -> np.ndarray:
#         # both tensors are (2 x T x N x S).
#         self._validate_shapes(main_nodes, between_nodes)
#         n_samples = main_nodes.shape[2]
#         padding = np.full(shape=(1, n_samples, self.num_strains), fill_value=-np.inf)
#         slice_0 = np.max(np.stack([
#             main_nodes[0],
#             np.concatenate([between_nodes[0], padding], axis=0),
#             np.concatenate([padding, between_nodes[0]], axis=0),
#         ]), axis=0)  # T x N x S
#         slice_1 = np.max(np.stack([
#             main_nodes[1],
#             np.concatenate([between_nodes[1], padding], axis=0),
#             np.concatenate([padding, between_nodes[1]], axis=0),
#         ]), axis=0)  # T x N x S
#         return np.greater(slice_0, slice_1)
#
#     def smooth_log_zeroes_of_gumbels(self, main_nodes: np.ndarray, between_nodes: np.ndarray, inv_temperature: float) -> np.ndarray:
#         """
#         @param main_nodes: (2 x T x N x S)
#         @param between_nodes: (2 x T-1 x N x S)
#         @param inv_temperature:
#         @return:
#         """
#         self._validate_shapes(main_nodes, between_nodes)
#         timeseries_pieces = []
#
#         # First timepoint
#         slice_0 = _smooth_max(
#             np.stack([main_nodes[0, 0], between_nodes[0, 0]], axis=0),
#             inv_temp=inv_temperature,
#             axis=0
#         )
#         slice_1 = _smooth_max(
#             np.stack([main_nodes[1, 0], between_nodes[1, 0]], axis=0),
#             inv_temp=inv_temperature,
#             axis=0
#         )
#         timeseries_pieces.append(
#             np.expand_dims(_smooth_log_p(np.stack([slice_0, slice_1], axis=0), inv_temperature), axis=0)
#         )
#
#         # In-between timepoints
#         if len(self.time_points) > 2:
#             slice_0 = _smooth_max(
#                 np.stack([main_nodes[0, 1:-1], between_nodes[0, 1:], between_nodes[0, :-1]], axis=0),
#                 inv_temp=inv_temperature,
#                 axis=0
#             )
#             slice_1 = _smooth_max(
#                 np.stack([main_nodes[1, 1:-1], between_nodes[1, 1:], between_nodes[1, :-1]], axis=0),
#                 inv_temp=inv_temperature,
#                 axis=0
#             )
#             timeseries_pieces.append(
#                 _smooth_log_p(np.stack([slice_0, slice_1], axis=0), inv_temperature)
#             )
#
#         # Last timepoint
#         slice_0 = _smooth_max(
#             np.stack([main_nodes[0, -1], between_nodes[0, -1]], axis=0),
#             inv_temp=inv_temperature,
#             axis=0
#         )
#         slice_1 = _smooth_max(
#             np.stack([main_nodes[1, -1], between_nodes[1, -1]], axis=0),
#             inv_temp=inv_temperature,
#             axis=0
#         )
#         timeseries_pieces.append(
#             np.expand_dims(_smooth_log_p(np.stack([slice_0, slice_1], axis=0), inv_temperature), axis=0)
#         )
#         return np.concatenate(timeseries_pieces, axis=0)

