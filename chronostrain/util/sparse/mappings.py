import torch
import torch_sparse

# ========= Youn's bbvi implementation


def normalize_sparse_2d(x_indices: torch.Tensor, x_values: torch.Tensor, x_rows: int, x_cols: int, dim: int) -> torch.Tensor:
    """
    Normalizes the specified dimension of a 2-d matrix (so that summing along that dimension returns a tensor of ones).
    The input matrix is assumed to be sparse in a single dimension only (either row-sparse or column-sparse)
    stored as a sparse_coo tensor.

    Note: only efficient if `dim` is over a sufficiently sparse dimension. (The 1-d row/col sum matrix will be
    converted to dense.)
    """
    # TODO: unit test, make sure torch.sparse.sum(normalize_sparse_2d(x, dim), dim) == [1, 1, ..., 1].
    sums = torch.sparse.sum(
        torch.sparse_coo_tensor(x_indices, x_values, (x_rows, x_cols)),  # this step might be slow? (creating a sparse coo)
        dim=dim
    ).to_dense()
    rescaled_values = x_values / sums[x_indices[1 - dim]]

    return torch.sparse_coo_tensor(
        indices=x_indices,
        values=rescaled_values,
        size=[x_rows, x_cols]
    )


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
# def exp(x):
#     if isinstance(x, CoalescedSparseTensor):
#         return x.exp()
#     else:
#         return torch.exp(x)
#
# def mul(left_factor, right_factor):
#     if isinstance(left_factor, CoalescedSparseTensor):
#         if isinstance(right_factor, CoalescedSparseTensor):
#             return left_factor.sparse_mul(right_factor)
#         return left_factor.dense_mul(right_factor)
#
#     if isinstance(left_factor, CoalescedSparseTensor):
#         return right_factor.dense_mul(left_factor)
#
#     return left_factor * right_factor


# def scalar_sum(x: CoalescedSparseTensor, scalar: Union[int, float]):
#     if isinstance(x, CoalescedSparseTensor):
#         return x.sparse_scalar_sum(scalar)
#     return x + scalar
#
#
# def row_hadamard(x: CoalescedSparseTensor, vec: torch.Tensor) -> Union[torch.Tensor, CoalescedSparseTensor]:
#     if isinstance(x, CoalescedSparseTensor):
#         return x.row_hadamard(vec)
#     return x * vec
#
#
# def column_normed_row_sum(x: CoalescedSparseTensor) -> Union[torch.Tensor, CoalescedSparseTensor]:
#     if isinstance(x, CoalescedSparseTensor):
#         return x.column_normed_row_sum()
#     return (x / x.sum(dim=0)[None, :]).sum(dim=1)
#
#
# def slice_cols(x: CoalescedSparseTensor, cols_to_keep: List[int]):
#     if isinstance(x, CoalescedSparseTensor):
#         return x.del_cols(cols_to_keep)
#     return x[:, cols_to_keep[0]]
