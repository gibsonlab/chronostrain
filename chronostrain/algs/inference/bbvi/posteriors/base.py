from abc import ABC, abstractmethod
from typing import List

import torch
from torch.nn import Parameter

from chronostrain.algs.inference.vi import AbstractPosterior


class AbstractReparametrizedPosterior(AbstractPosterior, ABC):
    def log_likelihood(self, samples: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def trainable_parameters(self) -> List[Parameter]:
        pass

    @abstractmethod
    def mean(self) -> torch.Tensor:
        pass

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        pass

    @abstractmethod
    def reparametrized_sample(self, num_samples: int) -> torch.Tensor:
        pass
