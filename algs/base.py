"""
 base.py
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
    fragment_space = model.bacteria_pop.get_fragment_space(model.read_length)
    return [
        [
            [np.exp(model.error_model.compute_log_likelihood(f, r)) for f in fragment_space.get_fragments()]
            for r in reads[k]
        ]
        for k in range(len(model.times))
    ]
