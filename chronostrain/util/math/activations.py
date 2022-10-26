"""
SparseMax (python-interfacing pytorch implementation)

Based on https://github.com/aced125/sparsemax
"""
import torch
from torch import Tensor
from torch.autograd import Function


def sparsemax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return Sparsemax.apply(x, dim)


def flatten_all_but_nth_dim(ctx, x: Tensor):
    """
    From https://github.com/aced125/sparsemax

    Flattens tensor in all but 1 chosen dimension.
    Saves necessary context for backward pass and unflattening.
    """
    # transpose batch and nth dim
    x = x.transpose(0, ctx.dim)

    # Get and save original size in context for backward pass
    original_size = x.size()
    ctx.original_size = original_size

    # Flatten all dimensions except nth dim
    x = x.reshape(x.size(0), -1)

    # Transpose flattened dimensions to 0th dim, nth dim to last dim
    return ctx, x.transpose(0, -1)


def unflatten_all_but_nth_dim(ctx, x: Tensor):
    """
    From https://github.com/aced125/sparsemax

    Unflattens tensor using necessary context
    """
    # Tranpose flattened dim to last dim, nth dim to 0th dim
    x = x.transpose(0, 1)

    # Reshape to original size
    x = x.reshape(ctx.original_size)

    # Swap batch dim and nth dim
    return ctx, x.transpose(0, ctx.dim)


# noinspection PyMethodOverriding
class Sparsemax(Function):
    @staticmethod
    def forward(ctx, input: Tensor, dim: int = -1):
        input_dim = input.dim()
        if input_dim <= dim or dim < -input_dim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of [-{input_dim}, {input_dim - 1}], but got {dim})"
            )

        # Save operating dimension to context
        ctx.needs_reshaping = input_dim > 2
        ctx.dim = dim

        if ctx.needs_reshaping:
            ctx, input = flatten_all_but_nth_dim(ctx, input)

        # Translate by max for numerical stability
        input = input - input.max(-1, keepdim=True).values.expand_as(input)

        zs = input.sort(-1, descending=True).values
        range = torch.arange(1, input.size()[-1] + 1)
        range = range.expand_as(input).to(input)

        # Determine sparsity of projection
        bound = 1 + range * zs
        is_gt = bound.gt(zs.cumsum(-1)).type(input.dtype)
        k = (is_gt * range).max(-1, keepdim=True).values

        # Compute threshold
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (zs_sparse.sum(-1, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        output = torch.max(torch.zeros_like(input), input - taus)

        # Save context
        ctx.save_for_backward(output)

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, output = unflatten_all_but_nth_dim(ctx, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, *_ = ctx.saved_tensors

        # Reshape if needed
        if ctx.needs_reshaping:
            ctx, grad_output = flatten_all_but_nth_dim(ctx, grad_output)

        # Compute gradient
        nonzeros = torch.ne(output, 0)
        num_nonzeros = nonzeros.sum(-1, keepdim=True)
        _sum = (grad_output[nonzeros]).sum(-1, keepdim=True) / num_nonzeros

        grad_input = torch.zeros(grad_output.shape, device=grad_output.device, dtype=grad_output.dtype)
        grad_input[nonzeros] = (grad_output - _sum.expand_as(grad_output))[nonzeros]

        if torch.sum(torch.isinf(grad_input)) > 0:
            print("Found inf in grad: {}".format(grad_input))
            exit(1)
        if torch.sum(num_nonzeros == 0) > 0:
            print("got nnz=0 in grad: {}".format(grad_input))
            exit(1)

        # Reshape back to original shape
        if ctx.needs_reshaping:
            ctx, grad_input = unflatten_all_but_nth_dim(ctx, grad_input)
        return grad_input, None
