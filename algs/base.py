"""
 base.py
 Contains implementations of the proposed algorithms.
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List
from model.generative import GenerativeModel
from model.reads import SequenceRead


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self, model: GenerativeModel, data: List[List[SequenceRead]]):
        self.model = model
        self.data = data

    @abstractmethod
    def solve(self, *args, **kwargs):
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
