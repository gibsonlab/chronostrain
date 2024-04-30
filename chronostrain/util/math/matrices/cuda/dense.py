import numpy as np
from numba import cuda
import math
import torch
from .constants import _DTYPE, _THREADS_PER_BLOCK
from ..cpu.dense import log_matmul_exp as cpu_like_torch_log_matmul_exp

from chronostrain.logging import create_logger
logger = create_logger(__name__)


_CUDA_WARNED = False


def log_matmul_exp(x: torch.Tensor, y: torch.Tensor):
    global _CUDA_WARNED
    assert x.device.type == "cuda" and y.device.type == "cuda"
    if x.requires_grad or y.requires_grad:
        if not _CUDA_WARNED:
            logger.debug("TODO: The custom implementation of cuda.log_matmul_exp doesn't support grad. "
                         "Defaulting to torch-native implementation; a C++ implementation should be provided instead.")
            _CUDA_WARNED = True

        return cpu_like_torch_log_matmul_exp(x, y)
    return cpu_like_torch_log_matmul_exp(x, y)

    # out = torch.empty((x.shape[0], y.shape[1]), dtype=x.dtype, device=x.device)
    #
    # # CUDA memory specification
    # threads_per_block = (_THREADS_PER_BLOCK, _THREADS_PER_BLOCK)
    # n_blocks_x = int(math.ceil(out.shape[0] / threads_per_block[0]))
    # n_blocks_y = int(math.ceil(out.shape[1] / threads_per_block[1]))
    # block_dims = (n_blocks_x, n_blocks_y)
    #
    # jit_log_matmul_exp[block_dims, threads_per_block](
    #     cuda.as_cuda_array(x),
    #     cuda.as_cuda_array(y),
    #     cuda.as_cuda_array(out)
    # )
    # return out


@cuda.jit
def jit_log_matmul_exp(A, B, C):
    """
    Perform matrix multiplication of C = A * B in log-space
    Each thread computes one element of the result matrix C
    Adapted from CUDA tutorial https://nyu-cds.github.io/python-numba/05-cuda/
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(_THREADS_PER_BLOCK, _THREADS_PER_BLOCK), dtype=_DTYPE)
    sB = cuda.shared.array(shape=(_THREADS_PER_BLOCK, _THREADS_PER_BLOCK), dtype=_DTYPE)

    # Absolute (row, column) position
    x, y = cuda.grid(2)

    # Relative position (inside sub-block sA, sB)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    log_sum_exp = -np.inf
    for i in range(int(math.ceil(A.shape[1] / _THREADS_PER_BLOCK))):
        num_j = min(_THREADS_PER_BLOCK, A.shape[1] - i * _THREADS_PER_BLOCK)

        # Preload read_frags into shared memory
        sA[tx, ty] = A[x, ty + i * _THREADS_PER_BLOCK]
        sB[tx, ty] = B[tx + i * _THREADS_PER_BLOCK, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Compute offset to use for logsumexp()
        offset = log_sum_exp
        for j in range(num_j):
            offset = max(offset, sA[tx, j] + sB[j, ty])

        # Compute summation of exponentials
        sum_exp = math.exp(log_sum_exp - offset)
        for j in range(num_j):
            sum_exp += math.exp(sA[tx, j] + sB[j, ty] - offset)

        # Compute logsumexp.
        log_sum_exp = math.log(sum_exp) + offset

        # Wait until all threads finish computing
        cuda.syncthreads()
    C[x, y] = log_sum_exp
