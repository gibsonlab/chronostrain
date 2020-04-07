"""
 model_solver.py
 Contains implementations of the proposed algorithms.
"""

import numpy as np
from abc import ABCMeta, abstractmethod


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
    return [
        [
            [np.exp(model.error_model.compute_log_likelihood(f, r)) for f in model.fragment_space]
            for r in reads[k]
        ]
        for k in range(len(model.times))
    ]
