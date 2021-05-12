import torch
from torch.nn.functional import softmax, pad


def multi_logit(x: torch.Tensor, dim) -> torch.Tensor:
    """
    Applies softmax along the specified dimension, after padding the secondary dimension on the right
    (columns of a Matrix, or entries of a vector) with a zero.
    If applied to a scalar tensor, effectively computes (p,q) = softmax(x, 0) = (logit(x), 1-logit(x)).
    """
    return softmax(
        pad(x, pad=[0, 1]),  # Add a column of zeros.
        dim=dim
    )


def normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Normalizes the specified dimension (so that summing along that dimension returns a tensor of ones).
    :param x:
    :param dim:
    :return:
    """
    return x / x.sum(dim=dim, keepdim=True)
