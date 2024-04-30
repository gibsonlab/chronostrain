from .sparse_tensor import SparseMatrix
from .sliceable import ColumnSectionedSparseMatrix, RowSectionedSparseMatrix, ADVIOptimizedSparseMatrix
from .ops import log_mm_exp, log_spspmm_exp

print("WARNING: using deperecated submodule `matrices`.")