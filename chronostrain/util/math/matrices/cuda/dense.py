from numba import cuda
import numpy as np
import math
import torch
from .constants import _DTYPE, _THREADS_PER_BLOCK

from chronostrain.logging import create_logger
logger = create_logger(__name__)


_CUDA_WARNED = False


def log_matmul_exp(x: torch.Tensor, y: torch.Tensor):
    global _CUDA_WARNED
    assert x.device.type == "cuda" and y.device.type == "cuda"
    if x.requires_grad or y.requires_grad:
        if not _CUDA_WARNED:
            logger.debug("TODO: The custom implementation of cuda.log_matmul_exp doesn't support grad. "
                         "Defaulting to torch-native CUDA implementation; a C++ implementation should be provided instead.")
            _CUDA_WARNED = True

        return torch_log_matmul_exp(x, y)

    out = torch.empty((x.shape[0], y.shape[1]), dtype=x.dtype, device=x.device)
    jit_log_matmul_exp(
        cuda.as_cuda_array(x),
        cuda.as_cuda_array(y),
        cuda.as_cuda_array(out)
    )
    return out


def torch_log_matmul_exp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(
        x.unsqueeze(2) + y.unsqueeze(0),
        dim=1,
        keepdim=False
    )


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
    tmp = 0.
    offset = -np.inf
    total = 0.
    for i in range(A.shape[1] // _THREADS_PER_BLOCK):  # i: index over blocks
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * _THREADS_PER_BLOCK]
        sB[tx, ty] = B[tx + i * _THREADS_PER_BLOCK, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Compute offset to use for logsumexp()
        for j in range(_THREADS_PER_BLOCK):
            # tmp += sA[tx, j] * sB[j, ty]
            offset = max(offset, sA[tx, j] + sB[j, ty])

        # Compute summation of exponentials
        for j in range(_THREADS_PER_BLOCK):
            total += math.exp(sA[tx, j] + sB[j, ty] - offset)

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = math.log(total) + offset
