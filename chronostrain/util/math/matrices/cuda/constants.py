import numba
import torch
from chronostrain.config import cfg


dtype_conversion = {
    torch.float: numba.float32,
    torch.float32: numba.float32,
    torch.float64: numba.float64,
    torch.double: numba.double,
    torch.uint8: numba.uint8,
    torch.int: numba.int32,
    torch.int8: numba.int8,
    torch.int16: numba.int16,
    torch.int32: numba.int32,
    torch.int64: numba.int64,
    torch.short: numba.int16,
    torch.long: numba.int64,
    torch.bool: numba.boolean
}


# ============= CONSTANTS
if cfg.torch_cfg.default_dtype not in dtype_conversion:
    raise RuntimeError(f"Can't translate torch dtype `{cfg.torch_cfg.default_dtype}` to numba dtype.")

_DTYPE = dtype_conversion[cfg.torch_cfg.default_dtype]


# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
_THREADS_PER_BLOCK = 16  # TODO make configurable
