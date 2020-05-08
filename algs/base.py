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

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# num_cores = multiprocessing.cpu_count()
num_cores = 1


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
def compute_read_likelihoods(
        model: GenerativeModel,
        reads: List[List[SequenceRead]],
        logarithm: bool,
        device) -> List[torch.Tensor]:
    fragment_space = model.bacteria_pop.get_fragment_space(model.read_length)

    start_time = current_time_millis()
    logger.debug("Computing read-fragment likelihoods...")

    def create_matrix(k):
        # Each is an (F x N) matrix,
        # where N is the number of reads in a given time point and F is the number of fragments.
        return torch.tensor([
            [
                model.error_model.compute_log_likelihood(f, r) if logarithm
                else math.exp(model.error_model.compute_log_likelihood(f, r))
                for r in reads[k]
            ] for f in fragment_space.get_fragments()
        ], device=device, dtype=torch.double)

    errors = Parallel(n_jobs=num_cores)(delayed(create_matrix)(k) for k in tqdm(range(len(model.times))))

    # ref: https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
    # TODO: Some 'future warnings' are being thrown about saving tensors (in the subprocesses).
    # TODO: Maybe find another parallelization alternative.
    logger.debug("Computed fragment errors in {} min.".format(millis_elapsed(start_time) / 60000))

    return errors
