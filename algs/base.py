"""
 base.py
 Contains implementations of the proposed algorithms.
"""

import torch
import math

from abc import ABCMeta, abstractmethod
from typing import List
from model.generative import GenerativeModel
from model.reads import SequenceRead

from util.io.logger import logger
from util.benchmarking import current_time_millis, millis_elapsed


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
def compute_read_likelihoods(model: GenerativeModel, reads: List[List[SequenceRead]], device) -> List[torch.Tensor]:
    fragment_space = model.bacteria_pop.get_fragment_space(model.read_length)

    start_time = current_time_millis()
    logger.info("Computing fragment errors...")
    errors = [
        # Each is an (F x N) matrix.
        torch.tensor([
            [
                math.exp(model.error_model.compute_log_likelihood(f, r))
                for r in reads[k]
            ] for f in fragment_space.get_fragments()
        ], device=device, dtype=torch.double)
        for k in range(len(model.times))
    ]

    # TODO https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
    logger.debug("Computed fragment errors in {} min.".format(millis_elapsed(start_time) / 60000))

    return errors
