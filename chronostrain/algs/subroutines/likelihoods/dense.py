import numpy as np
import torch
from typing import List
from joblib import Parallel, delayed

from chronostrain.model import Fragment, SequenceRead
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from chronostrain.model.generative import GenerativeModel
from chronostrain.util.math import log_mm_exp

from .base import DataLikelihoods, AbstractLogLikelihoodComputer
from ..cache import ReadsPopulationCache

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class DenseDataLikelihoods(DataLikelihoods):
    def _likelihood_computer(self) -> AbstractLogLikelihoodComputer:
        return DenseLogLikelihoodComputer(self.model, self.data)

    def conditional_likelihood(self, X: torch.Tensor, inf_fill: float = -100000) -> float:
        y = torch.softmax(X, dim=1)
        # Calculation is sigma(X) @ W @ E.
        total_ll = 0.
        for t in range(self.model.num_times()):
            # (1 x S) * (S x F) * (F x N) = (1 x N)
            read_likelihoods = log_mm_exp(
                y[t].log().view(1, -1),
                log_mm_exp(
                    self.model.fragment_frequencies_dense.t(),
                    self.matrices[t]
                )
            )
            read_likelihoods[torch.isinf(read_likelihoods)] = -inf_fill
            total_ll += read_likelihoods.sum()
        return total_ll


class DenseLogLikelihoodComputer(AbstractLogLikelihoodComputer):

    def __init__(self, model: GenerativeModel, reads: TimeSeriesReads, num_cores: int = 1):
        self.num_cores = num_cores
        super().__init__(model, reads)

    def compute_matrix_single_timepoint(self, t_idx: int) -> List[List[float]]:
        """
        For the specified time point, evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :param t_idx: The time point index to run this function on.
        :returns: The array of likelihoods, stored as a length-F list of length-N_t lists.
        """
        ans = [
            [
                self.compute_forward_reverse_log_likelihood(frag, read)
                for read in self.reads[t_idx]
            ]
            for frag in self.model.fragments.get_fragments()
        ]
        return ans

    def compute_forward_reverse_log_likelihood(self, frag: Fragment, read: SequenceRead) -> float:
        """
        Computes log(p), where p = 0.5 * P(read | frag, forward) + 0.5 * P(read | frag, reverse)
        which assumes an equal likelihood of sampling from the forward and reverse strands in sequencing.

        This computation assumes no indel errors. The specified fragment's length must equal the read's length.
        """
        forward_ll = self.model.error_model.compute_log_likelihood(frag, read, read_reverse_complemented=False)
        reverse_ll = self.model.error_model.compute_log_likelihood(frag, read, read_reverse_complemented=True)
        log2 = np.log(2)
        return np.log(np.exp(forward_ll - log2) + np.exp(reverse_ll - log2))

    def compute_likelihood_tensors(self) -> List[torch.Tensor]:
        logger.debug("Computing read-fragment likelihoods...")
        cache = ReadsPopulationCache(self.reads, self.model.bacteria_pop)

        jobs = [
            {
                "filename": "log_likelihoods_{}.pkl".format(t_idx),
                "fn": lambda t: self.compute_matrix_single_timepoint(t),
                "args": [],
                "kwargs": {"t": t_idx}
            }
            for t_idx in range(self.model.num_times())
        ]

        parallel = (self.num_cores > 1)
        if parallel:
            logger.debug("Computing read likelihoods with parallel pool size = {}.".format(self.num_cores))

            log_likelihoods_output = Parallel(n_jobs=self.num_cores)(
                delayed(cache.call)(cache_kwargs)
                for cache_kwargs in jobs
            )

            log_likelihoods_tensors = [
                torch.tensor(ll_array, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
                for ll_array in log_likelihoods_output
            ]
        else:
            log_likelihoods_tensors = [
                torch.tensor(cache.call(**cache_kwargs), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
                for cache_kwargs in jobs
            ]

        return log_likelihoods_tensors
