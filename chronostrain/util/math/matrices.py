import torch


@torch.jit.script
def logmatmulexp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((x.shape[0], y.shape[1]), device=x.device)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i, j] = torch.logsumexp(x[i, :] + y[:, j], dim=0)
    return z
