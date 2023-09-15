from .constants import GENERIC_GRAD_TYPE, GENERIC_SAMPLE_TYPE, GENERIC_PARAM_TYPE
from .advi import AbstractADVI, AbstractADVISolver
from .posterior import AbstractPosterior, AbstractReparametrizedPosterior
from .util import divide_columns_into_batches, divide_columns_into_batches_sparse, \
    log_dot_exp, log_mv_exp, log_mm_exp
