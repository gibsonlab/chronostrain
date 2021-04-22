"""
 base.py
 Contains implementations of the proposed algorithms.
"""

import torch
from typing import List

from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from tqdm import tqdm

from . import logger
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel
from chronostrain.util.data_cache import CachedComputation, CacheTag
from chronostrain.util.benchmarking import current_time_millis, millis_elapsed


class AbstractModelSolver(metaclass=ABCMeta):
    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 cache_tag: CacheTag):
        self.model = model
        self.data = data
        self.cache_tag = cache_tag

        # Not sure which we will need. Use lazy initialization.
        self.read_likelihoods_loaded = False
        self.read_likelihoods_tensors: List[torch.Tensor] = []

        self.read_log_likelihoods_loaded = False
        self.read_log_likelihoods_tensors: List[torch.Tensor] = []

    @abstractmethod
    def solve(self, *args, **kwargs):
        pass

    @property
    def read_likelihoods(self) -> List[torch.Tensor]:
        if not self.read_likelihoods_loaded:
            log_likelihoods = CachedComputation(compute_read_log_likelihoods, cache_tag=self.cache_tag).call(
                "read_log_likelihoods.pkl",
                model=self.model,
                reads=self.data
            )
            self.read_likelihoods_tensors = [
                torch.exp(ll_tensor).to(cfg.torch_cfg.device)
                for ll_tensor in log_likelihoods
            ]
            self.read_likelihoods_loaded = True
        return self.read_likelihoods_tensors

    @property
    def read_log_likelihoods(self) -> List[torch.Tensor]:
        if not self.read_log_likelihoods_loaded:
            log_likelihoods = CachedComputation(compute_read_log_likelihoods, cache_tag=self.cache_tag).call(
                "read_log_likelihoods.pkl",
                model=self.model,
                reads=self.data
            )
            self.read_log_likelihoods_tensors = [
                ll_tensor.to(cfg.torch_cfg.device)
                for ll_tensor in log_likelihoods
            ]
            self.read_log_likelihoods_loaded = True
        return self.read_log_likelihoods_tensors


# ===================================================================
# ========================= Helper functions ========================
# ===================================================================

# Helper function
def compute_read_log_likelihoods(model: GenerativeModel, reads: TimeSeriesReads) -> List[torch.Tensor]:
    """
    Returns a list of (F x N) tensors, each containing the time-t read likelihoods.
    """
    fragment_space = model.get_fragment_space()

    start_time = current_time_millis()
    logger.debug("Computing read-fragment likelihoods...")

    def create_matrix(k):
        """
        For the specified time point (t = t_k), evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :param k: The time point index to run this function on.
        :returns: The array of likelihoods, stored as a length-F list of length-N_t lists.
        """
        start_t = current_time_millis()
        ans = [
            [
                model.error_model.compute_log_likelihood(frag, read)
                for read in reads[k]
            ] for frag in fragment_space.get_fragments()
        ]
        return ans

    parallel = (cfg.model_cfg.num_cores > 1)
    if parallel:
        logger.debug("Computing read likelihoods with parallel pool size = {}.".format(cfg.model_cfg.num_cores))
        log_likelihoods_output = Parallel(
            n_jobs=cfg.model_cfg.num_cores
        )(
            delayed(create_matrix)(k)
            for k in tqdm(range(len(model.times)), desc="Read Prob.")
        )
        log_likelihoods_tensors = [
            torch.tensor(ll_array, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
            for ll_array in log_likelihoods_output
        ]
    else:
        log_likelihoods_tensors = [
            torch.tensor(create_matrix(k), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
            for k in tqdm(range(len(model.times)))
        ]
    logger.debug("Computed fragment errors in {:1f} min.".format(millis_elapsed(start_time) / 60000))

    return log_likelihoods_tensors
