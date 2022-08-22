from torch import Tensor
from . import cpu
from . import cuda

from chronostrain.config import cfg
from .sliceable import ColumnSectionedSparseMatrix, RowSectionedSparseMatrix

from chronostrain.logging import create_logger
logger = create_logger(__name__)


def log_mm_exp(x: Tensor, y: Tensor) -> Tensor:
    # Choose the proper interface (cpu vs cuda).
    if cfg.torch_cfg.device.type == "cpu":
        return cpu.log_matmul_exp(x, y)
    elif cfg.torch_cfg.device.type == "cuda":
        return cuda.log_matmul_exp(x, y)
    else:
        raise RuntimeError(f"Unknown device name `{cfg.torch_cfg.device.type}`.")


logger.debug("TODO: cuda.log_spspmm_exp has not yet been implemented; by default, it will revert to cpu implementation.")
def log_spspmm_exp(x: ColumnSectionedSparseMatrix, y: RowSectionedSparseMatrix) -> Tensor:
    # Choose the proper interface (cpu vs cuda).
    if cfg.torch_cfg.device.type == "cpu":
        return cpu.log_spspmm_exp(x, y)
    elif cfg.torch_cfg.device.type == "cuda":
        # return cuda.log_spspmm_exp(x, y)  # TODO CUDA
        return cpu.log_spspmm_exp(x, y)
    else:
        raise RuntimeError(f"Unknown device name `{cfg.torch_cfg.device.type}`.")