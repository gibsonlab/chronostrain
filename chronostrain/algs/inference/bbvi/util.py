import torch
import torch.nn.functional
from chronostrain.config import cfg


def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x - torch.logsumexp(x, dim=dim, keepdim=True)


def log_spherical(x: torch.Tensor, dim: int, eps=1e-30, keepdim: bool = True) -> torch.Tensor:
    # x_samples: (T x N x S) tensor.
    square = torch.pow(x, 2) + eps
    return torch.log(square) - torch.log(square.sum(dim=dim, keepdim=keepdim))


def log_taylor(x: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
    exp_taylor = 1 + x + 0.5 * torch.pow(x, 2)
    return torch.log(exp_taylor) - torch.log(exp_taylor.sum(dim=dim, keepdim=keepdim))


def divide_columns_into_batches(x: torch.Tensor, batch_size: int):
    permutation = torch.randperm(x.shape[1], device=cfg.torch_cfg.device)
    for i in range(0, x.shape[1], batch_size):
        indices = permutation[i:i+batch_size]
        yield x[:, indices]
