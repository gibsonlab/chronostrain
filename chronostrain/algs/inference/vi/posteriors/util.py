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


class TrilLinear(torch.nn.Module):
    """
    Represents transformation by a lower triangular (with strictly positive diagonal entries) matrix, with optional
    bias term.
    Used for reparametrizing a Gaussian by multiplying a standard normal by the Cholesky factor of the target
    covariance matrix.
    """
    def __init__(self, n_features: int, bias: bool, device=None, dtype=None):
        super().__init__()
        nnz = n_features * (n_features - 1) // 2
        self.tril_weights = torch.nn.Parameter(torch.zeros(nnz, device=device, dtype=dtype))
        self.diag_weights = torch.nn.Parameter(torch.zeros(n_features, device=device, dtype=dtype))
        self.tril_ind = torch.tril_indices(n_features, n_features, -1)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(n_features, device=device, dtype=dtype))
        else:
            self.bias = torch.zeros(n_features, device=device, dtype=dtype)
        self.n_features = n_features
        self.device = device
        self.dtype = dtype

    @property
    def weight(self) -> torch.Tensor:
        A = torch.zeros(size=(self.n_features, self.n_features), device=self.device, dtype=self.dtype)
        tril_r, tril_c = self.tril_ind
        diag_r = torch.arange(0, self.n_features)
        A[tril_r, tril_c] = self.tril_weights
        A[diag_r, diag_r] = torch.exp(self.diag_weights)
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.linear(x, self.weight, self.bias)


class BatchedTrilLinear(torch.nn.Module):
    """
    Applies a batched linear transformation to the incoming data by block-diagonal transformation.
    Amounts to a single (N x Bp) @ (Bp x Bq) -> (BN x q) multiplication
    where N is the number of samples in the input, and B is the number of batches
    (the input must be reshaped accordingly beforehand.)

    weight: the learnable weights of the module of shape (in_features) x (out_features).
    """

    def __init__(self, in_features: int, out_features: int, n_batches: int, device=None, dtype=None):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.empty(n_batches, out_features, in_features, device=device, dtype=dtype)
        )

    @property
    def cholesky_part(self) -> torch.Tensor:
        w = torch.block_diag(*self.weights)
        return torch.tril(w, diagonal=-1) + torch.diag(torch.exp(torch.diag(w)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.cholesky_part)
