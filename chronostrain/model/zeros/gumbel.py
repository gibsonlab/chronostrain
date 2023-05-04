from typing import List
import torch
from torch import Tensor

from chronostrain.config import cfg


def _expect_tensor_shape(x: Tensor, name: str, shape: List[int]):
    if (len(x.shape) != len(shape)) or (list(x.shape) != shape):
        raise ValueError("Tensor `{}` must be of size {}. Got: {}".format(
            name,
            shape,
            x.shape
        ))


def _smooth_argmax(logits: Tensor, inv_temp: Tensor, dim: int) -> Tensor:
    return torch.softmax(inv_temp * logits, dim=dim)


def _smooth_log_argmax(logits: Tensor, inv_temp: Tensor, dim: int) -> Tensor:
    """
    Logits are assumed to be stacked along dimension 0.
    """
    return torch.log_softmax(inv_temp * logits, dim=dim)


def _smooth_log_p(logits: Tensor, inv_temp: Tensor) -> Tensor:
    return inv_temp * logits[0] - torch.logsumexp(logits * inv_temp, dim=0)


def _smooth_max(x: Tensor, inv_temp: Tensor, dim: int) -> Tensor:
    """
    x is assumed to be stacked along dimension 0.
    """
    return torch.nansum(torch.softmax(inv_temp * x, dim=dim) * x, dim=dim)
    # return torch.logsumexp(
    #     log_softmax(inv_temp * x, dim) + torch.log(x),
    #     dim=dim
    # )


class PopulationGlobalZeros(object):
    def __init__(self, num_strains: int):
        self.num_strains = num_strains
        self.gumbel_rv = torch.distributions.gumbel.Gumbel(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

        import numpy as np
        self.p = np.power(0.5, num_strains)

    def log_likelihood(self, zeros: Tensor) -> Tensor:
        """
        @param zeros: An (N x S) tensor of zeros or ones. (likelihood won't depend on smoothness)
        @return: a length-N tensor of likelihoods, one per sample.
        """
        # likelihood actually doesn't depend on the actual zeros/ones since prior is Bernoulli(0.5),
        # conditioned on not all being zero.
        return torch.full(zeros.size(), self.p, device=zeros.device)


class PopulationLocalZeros(object):
    def __init__(self, time_points: List[float], num_strains: int):
        self.time_points = time_points
        self.num_strains = num_strains
        self.gumbel_rv = torch.distributions.gumbel.Gumbel(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def _validate_shapes(self, main_nodes: Tensor, between_nodes: Tensor):
        n_samples = main_nodes.shape[2]
        _expect_tensor_shape(main_nodes, "main_nodes", [2, len(self.time_points), n_samples, self.num_strains])
        _expect_tensor_shape(between_nodes, "between_nodes", [2, len(self.time_points) - 1, n_samples, self.num_strains])

    def log_likelihood(self, main_nodes: Tensor, between_nodes: Tensor) -> Tensor:
        # both tensors are (2 x T x N x S).
        self._validate_shapes(main_nodes, between_nodes)
        return self.gumbel_rv.log_prob(
            main_nodes
        ).sum(dim=0).sum(dim=0).sum(dim=-1) + self.gumbel_rv.log_prob(
            between_nodes
        ).sum(dim=0).sum(dim=0).sum(dim=-1)

    def zeroes_of_gumbels(self, main_nodes: Tensor, between_nodes: Tensor) -> Tensor:
        # both tensors are (2 x T x N x S).
        self._validate_shapes(main_nodes, between_nodes)
        n_samples = main_nodes.shape[2]
        padding = torch.full((1, n_samples, self.num_strains), -torch.inf, device=main_nodes.device)
        slice_0 = torch.max(torch.stack([
            main_nodes[0],
            torch.concat([between_nodes[0], padding], dim=0),
            torch.concat([padding, between_nodes[0]], dim=0),
        ]), dim=0).values  # T x N x S
        slice_1 = torch.max(torch.stack([
            main_nodes[1],
            torch.concat([between_nodes[1], padding], dim=0),
            torch.concat([padding, between_nodes[1]], dim=0),
        ]), dim=0).values  # T x N x S
        return torch.gt(slice_0, slice_1).int()

    # def smooth_zeroes_of_gumbels(self, main_nodes: Tensor, between_nodes: Tensor, inv_temperature: float) -> Tensor:
    #     self._validate_shapes(main_nodes, between_nodes)
    #     inv_temperature = torch.tensor(inv_temperature, device=main_nodes.device)
    #     n_samples = main_nodes.shape[2]
    #     padding = torch.full((1, n_samples, self.num_strains), -torch.inf, device=main_nodes.device)
    #     slice_0 = log_softmax(inv_temperature * torch.stack([
    #         main_nodes[0],
    #         torch.concat([between_nodes[0], padding], dim=0),  # TODO fix case of this being multiple samples.
    #         torch.concat([padding, between_nodes[0]], dim=0),
    #     ]), dim=0)  # T x N x S
    #     slice_1 = log_softmax(inv_temperature * torch.stack([
    #         main_nodes[1],
    #         torch.concat([between_nodes[1], padding], dim=0),
    #         torch.concat([padding, between_nodes[1]], dim=0),
    #     ]), dim=0)  # T x N x S
    #     return _smooth_max(slice_0, slice_1, inv_temperature)[0]

    def smooth_log_zeroes_of_gumbels(self, main_nodes: Tensor, between_nodes: Tensor, inv_temperature: float) -> Tensor:
        """
        @param main_nodes: (2 x T x N x S)
        @param between_nodes: (2 x T-1 x N x S)
        @param inv_temperature:
        @return:
        """
        self._validate_shapes(main_nodes, between_nodes)
        inv_temperature = torch.tensor(inv_temperature, device=main_nodes.device)
        # return _smooth_log_argmax(
        #     main_nodes,
        #     dim=0,
        #     inv_temp=inv_temperature
        # )[0]

        timeseries_pieces = []

        # First timepoint
        slice_0 = _smooth_max(
            torch.stack([main_nodes[0, 0], between_nodes[0, 0]], dim=0),
            inv_temp=inv_temperature,
            dim=0
        )
        slice_1 = _smooth_max(
            torch.stack([main_nodes[1, 0], between_nodes[1, 0]], dim=0),
            inv_temp=inv_temperature,
            dim=0
        )
        timeseries_pieces.append(
            _smooth_log_p(torch.stack([slice_0, slice_1], dim=0), inv_temperature).unsqueeze(0)
        )

        # In-between timepoints
        if len(self.time_points) > 2:
            slice_0 = _smooth_max(
                torch.stack([main_nodes[0, 1:-1], between_nodes[0, 1:], between_nodes[0, :-1]], dim=0),
                inv_temp=inv_temperature,
                dim=0
            )
            slice_1 = _smooth_max(
                torch.stack([main_nodes[1, 1:-1], between_nodes[1, 1:], between_nodes[1, :-1]], dim=0),
                inv_temp=inv_temperature,
                dim=0
            )
            timeseries_pieces.append(
                _smooth_log_p(torch.stack([slice_0, slice_1], dim=0), inv_temperature)
            )

        # Last timepoint
        slice_0 = _smooth_max(
            torch.stack([main_nodes[0, -1], between_nodes[0, -1]], dim=0),
            inv_temp=inv_temperature,
            dim=0
        )
        slice_1 = _smooth_max(
            torch.stack([main_nodes[1, -1], between_nodes[1, -1]], dim=0),
            inv_temp=inv_temperature,
            dim=0
        )
        timeseries_pieces.append(
            _smooth_log_p(torch.stack([slice_0, slice_1], dim=0), inv_temperature).unsqueeze(0)
        )
        return torch.concat(timeseries_pieces, dim=0)

