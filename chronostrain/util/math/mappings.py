import torch
import torch_sparse
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

# ========= Youn's bbvi implementation

def normalize_sparse_2d(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Normalizes the specified dimension of a 2-d matrix (so that summing along that dimension returns a tensor of ones).
    The input matrix is assumed to be sparse in a single dimension only (either row-sparse or column-sparse)
    stored as a sparse_coo tensor.

    Note: only efficient if `dim` is over a sufficiently sparse dimension. (The 1-d row/col sum matrix will be
    converted to dense.)
    """
    # TODO: unit test, make sure torch.sparse.sum(normalize_sparse_2d(x, dim), dim) == [1, 1, ..., 1].
    sums = torch.sparse.sum(x, dim=dim).to_dense()
    rescaled_values = x.values() / sums[x.indices()[1 - dim]]

    return torch.sparse_coo_tensor(
        indices=x.indices(),
        values=rescaled_values,
        size=x.size(),
        dtype=x.dtype,
        device=x.device
    )


@torch.jit.script
def sparse_tensor_slice(indices: torch.Tensor, values: torch.Tensor, dim: int, idx: int):
    """
    Returns the dimension `dim`, index `idx` slice of the sparse matrix.
    :param indices:
    :param values:
    :param dim:
    :param idx:
    :return:
    """
    matching_entries = (indices[dim] == idx)
    return indices[:, matching_entries], values[matching_entries]


@torch.jit.script
def sparse_to_dense(index: torch.Tensor, value: torch.Tensor, rows: int, columns: int):
    x = torch.zeros((rows, columns))
    x[index[0], index[1]] = value
    return x


def spspmm(x: torch.Tensor, y: torch.Tensor):
    if x.size()[1] != y.size()[0]:
        raise ValueError("# of columns of x do not match # of rows of y.")
    i, v = torch_sparse.spspmm(
        indexA=x.indices(),
        valueA=x.values(),
        indexB=y.indices(),
        valueB=y.values(),
        m=x.size()[0],
        k=x.size()[1],
        n=y.size()[1],
        coalesced=False
    )
    return torch.sparse_coo_tensor(
        indices=i, values=v, size=torch.Size([x.size()[0], y.size()[1]]),
        dtype=x.dtype, device=x.device
    )


# =============== Zack's implementation
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