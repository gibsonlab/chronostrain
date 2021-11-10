from abc import abstractmethod, ABCMeta
from typing import List

import torch

from chronostrain.model import GenerativeModel
from chronostrain.model.io import TimeSeriesReads

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
            ll_tensor for ll_tensor in log_likelihoods_tensors
        ]

    @abstractmethod
    def _likelihood_computer(self) -> 'AbstractLogLikelihoodComputer':
        raise NotImplementedError()

    @abstractmethod
    def conditional_likelihood(self, X: torch.Tensor) -> float:
        """
        Computes the conditional data likelihood p(Data | X).
        :param X: The (T x S) tensor of latent abundance representations.
        :return:
        """


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
