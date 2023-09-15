from typing import *
import jax.numpy as np


GENERIC_PARAM_TYPE = Dict[str, np.ndarray]
GENERIC_GRAD_TYPE = GENERIC_PARAM_TYPE  # the two types usually tend to match.
GENERIC_SAMPLE_TYPE = Union[Dict[Any, np.ndarray], np.ndarray]




