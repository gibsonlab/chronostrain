import torch
from torch.nn.functional import softmax, pad
from chronostrain.util.sparse.sparse_tensor import coalesced_sparse_tensor

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

def exp(x):
    if type(x) == coalesced_sparse_tensor:
        return x.exp()
    else:
        return torch.exp(x)

def mul(left_factor, right_factor):
    if type(left_factor) == coalesced_sparse_tensor:
        if type(right_factor) == coalesced_sparse_tensor:
            return left_factor.sparse_mul(right_factor)
        return left_factor.dense_mul(right_factor)

    if type(right_factor) == coalesced_sparse_tensor:
        return right_factor.dense_mul(left_factor)

    return left_factor * right_factor

def scalar_sum(x, scalar):
    if type(x) == coalesced_sparse_tensor:
        return x.sparse_scalar_sum(scalar)
    return x + scalar

def row_hadamard(x, vec):
    if type(x) == coalesced_sparse_tensor:
        return x.row_hadamard(vec)
    return x * vec

def column_normed_row_sum(x):
    if type(x) == coalesced_sparse_tensor:
        return x.column_normed_row_sum()
    return (x / x.sum(dim=0)[None, :]).sum(dim=1)

def slice_cols(x, cols_to_keep: torch.tensor):
    if type(x) == coalesced_sparse_tensor:
        return x.del_cols(cols_to_keep)
    return x[:,cols_to_keep[0]]