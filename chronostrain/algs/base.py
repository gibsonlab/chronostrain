"""
 base.py
 Contains implementations of the proposed algorithms.
"""

import torch
import math

from abc import ABCMeta, abstractmethod
from typing import List

from chronostrain.util.logger import logger
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.reads import SequenceRead
from chronostrain.util.data_cache import CachedComputation
from chronostrain.util.benchmarking import current_time_millis, millis_elapsed

from joblib import Parallel, delayed
from tqdm import tqdm


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self,
                 model: GenerativeModel,
                 data: List[List[SequenceRead]],
                 cache_tag: str):
        self.model = model
        self.data = data
        self.cache_tag = cache_tag

        # Not sure which we will need. Use lazy initialization.
        self.read_likelihoods_tensors: List[torch.Tensor] = None
        self.read_log_likelihoods_tensors: List[torch.Tensor] = None

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

    @property
    def read_likelihoods(self) -> List[torch.Tensor]:
        if self.read_likelihoods_tensors is None:
            log_likelihoods = CachedComputation(compute_read_likelihoods, cache_tag=self.cache_tag).call(
                "read_log_likelihoods.pkl",
                model=self.model,
                reads=self.data,
                logarithm=False
            )
            self.read_likelihoods_tensors = [
                torch.exp(ll_tensor) for ll_tensor in log_likelihoods
            ]
        return self.read_likelihoods_tensors

    @property
    def read_log_likelihoods(self) -> List[torch.Tensor]:
        if self.read_log_likelihoods_tensors is None:
            self.read_log_likelihoods_tensors = CachedComputation(compute_read_likelihoods, cache_tag=self.cache_tag).call(
                "read_log_likelihoods.pkl",
                model=self.model,
                reads=self.data,
                logarithm=True
            )
        return self.read_log_likelihoods_tensors


# ===================================================================
# ========================= Helper functions ========================
# ===================================================================

# Helper function
def compute_read_likelihoods(
        model: GenerativeModel,
        reads: List[List[SequenceRead]],
        logarithm: bool) -> List[torch.Tensor]:
    """
    Returns a list of (F x N) tensors, each containing the time-t read likelihoods.
    """
    fragment_space = model.get_fragment_space()

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
        ], device=cfg.torch_cfg.device, dtype=torch.double)

    parallel = False
    if parallel:
        errors = Parallel(n_jobs=cfg.model_cfg.num_cores)(delayed(create_matrix)(k) for k in tqdm(range(len(model.times))))
        # ref: https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
        # TODO: Some 'future warnings' are being thrown about saving tensors (in the subprocesses).
        # TODO: Maybe find another parallelization alternative.
    else:
        errors = [
            create_matrix(k) for k in tqdm(range(len(model.times)))
        ]
    logger.debug("Computed fragment errors in {} min.".format(millis_elapsed(start_time) / 60000))

    return errors
