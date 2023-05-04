import torch


def log_spherical(x: torch.Tensor, dim: int, eps=1e-30, keepdim: bool = True) -> torch.Tensor:
    square = torch.pow(x, 2) + eps
    return torch.log(square) - torch.log(square.sum(dim=dim, keepdim=keepdim))


def log_taylor(x: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
    exp_taylor = 1 + x + 0.5 * torch.pow(x, 2)
    return torch.log(exp_taylor) - torch.log(exp_taylor.sum(dim=dim, keepdim=keepdim))
