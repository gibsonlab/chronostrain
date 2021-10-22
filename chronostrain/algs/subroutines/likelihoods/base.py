from abc import abstractmethod, ABCMeta
from typing import List

import torch

from chronostrain.model import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.sparse import SparseMatrix

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class DataLikelihoods(object):
    def __init__(
            self,
            model: GenerativeModel,
            data: TimeSeriesReads,
            read_likelihood_lower_bound: float = 1e-30
    ):
        """
        :param model:
        :param data:
        :param read_likelihood_lower_bound: Thresholds reads by this likelihood value.
            For each read, if the sum of all fragment-read likelihoods over all fragments does not exceed this value,
            the read is trimmed from the matrix (at the particular timepoint which it belongs to).
            (Note: passing '0' for this argument is the same as bypassing this filter.)
        """
        self.model = model
        self.data = data
        self.read_likelihood_lower_bound = read_likelihood_lower_bound

        log_likelihoods_tensors = self._likelihood_computer().compute_likelihood_tensors()
        self.matrices = [
            ll_tensor.exp() for ll_tensor in log_likelihoods_tensors
        ]
        self.retained_indices = self._trim()

    @abstractmethod
    def _likelihood_computer(self) -> 'AbstractLogLikelihoodComputer':
        raise NotImplementedError()

    def _trim(self) -> List[List[int]]:
        """
        Trims the likelihood matrices using the specified lower bound. Reads are removed if there are no fragments
        with likelihood greater than the lower bound. (This should ideally not happen if a stringent alignment-based
        filter was applied.)

        :return: List of the index of kept reads (exceeding the lower bound threshold).
        """
        read_indices = []
        for t_idx in range(self.model.num_times()):
            read_likelihoods_t = self.matrices[t_idx]
            sums = read_likelihoods_t.sum(dim=0)

            zero_indices = {i.item() for i in torch.where(sums <= self.read_likelihood_lower_bound)[0]}
            if len(zero_indices) > 0:
                logger.warning("[t = {}] Discarding {} of {} reads with overall likelihood < {}: {}".format(
                    self.model.times[t_idx],
                    len(zero_indices),
                    len(sums),
                    self.read_likelihood_lower_bound,
                    ",".join(str(read_idx) for read_idx in zero_indices)
                ))

                leftover_indices = [
                    read_idx
                    for read_idx in range(len(self.data[t_idx]))
                    if read_idx not in zero_indices
                ]
                read_indices.append(leftover_indices)

                if isinstance(read_likelihoods_t, SparseMatrix):
                    self.matrices[t_idx] = read_likelihoods_t.slice_columns(
                        leftover_indices
                    )
                else:
                    self.matrices[t_idx] = read_likelihoods_t[:, leftover_indices]
            else:
                read_indices.append(list(range(len(self.data[t_idx]))))
        return read_indices


class AbstractLogLikelihoodComputer(metaclass=ABCMeta):
    def __init__(self, model: GenerativeModel, reads: TimeSeriesReads):
        self.model = model
        self.reads = reads

    @abstractmethod
    def compute_likelihood_tensors(self) -> List[torch.Tensor]:
        """
        For each time point, evaluate the (F x N_t) array of fragment-to-read likelihoods.

        :returns: The array of likelihood tensors, indexed by timepoint indices.
        """
        pass
