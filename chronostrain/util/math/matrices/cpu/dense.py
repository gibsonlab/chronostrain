import torch


def log_matmul_exp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(
        x.unsqueeze(2) + y.unsqueeze(0),
        dim=1,
        keepdim=False
    )
