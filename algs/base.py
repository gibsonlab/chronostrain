"""
 base.py
 Contains implementations of the proposed algorithms.
"""

import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List
from model.generative import GenerativeModel
from model.reads import SequenceRead

from util.logger import logger
import time

import torch

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

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
def compute_frag_errors(model: GenerativeModel, reads: List[List[SequenceRead]]) -> torch.Tensor:
    fragment_space = model.bacteria_pop.get_fragment_space(model.read_length)

    start_time = time.time()
    logger.info("Computing fragment errors...")
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

    return torch.tensor(errors, device=default_device)
