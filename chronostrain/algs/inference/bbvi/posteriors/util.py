import torch
from torch.nn import functional


def init_diag(x: torch.Tensor, scale: float):
    """
    A reimplementation of torch.nn.init.eye_, so that the diagonal is an arbitrary value.
    :param x:
    :param scale:
    :return:
    """
    with torch.no_grad():
        torch.eye(*x.shape, out=x, requires_grad=x.requires_grad)
        torch.mul(x, scale, out=x)
    return x


class TrilLinear(torch.nn.Linear):
    """
    Represents transformation by a lower triangular (with strictly positive diagonal entries) matrix, with optional
    bias term.
    Used for reparametrizing a Gaussian by multiplying a standard normal by the Cholesky factor of the target
    covariance matrix.
    """
    def __init__(self, n_features: int, bias: bool, device=None, dtype=None):
        super().__init__(
            in_features=n_features,
            out_features=n_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.n_features = n_features

    @property
    def cholesky_part(self) -> torch.Tensor:
        return torch.tril(self.weight, diagonal=-1) + torch.diag(torch.exp(torch.diag(self.weight)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.linear(x, self.cholesky_part, self.bias)
