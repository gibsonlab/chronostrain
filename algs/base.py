"""
 base.py
 Contains implementations of the proposed algorithms.
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from util.logger import logger
import time

class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self, generative_model, data):
        self.model = generative_model
        self.data = data

    @abstractmethod
    def solve(self):
        pass


# ===================================================================
# ========================= Helper functions ========================
# ===================================================================

# Helper function for both EM and VI
def compute_frag_errors(model, reads):
    logger.info("Computing fragment errors...")
    fragment_space = model.bacteria_pop.get_fragment_space(model.read_length)

    start_time = time.time()
    errors = [
        [
            [np.exp(model.error_model.compute_log_likelihood(f, r)) for f in fragment_space.get_fragments()]
            for r in reads[k]
        ]
        for k in range(len(model.times))
    ]  # TODO: This is taking a very long time. Compute in parallel?

    elapsed_time = time.time() - start_time
    # https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
    logger.info("Computed fragment errors! Time taken to calculate fragment errors: " + str(elapsed_time))

    return errors
