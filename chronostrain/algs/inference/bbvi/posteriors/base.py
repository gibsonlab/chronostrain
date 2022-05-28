from abc import ABC, abstractmethod
from typing import List

import torch
from torch.nn import Parameter

from chronostrain.algs.inference.vi import AbstractPosterior


class AbstractReparametrizedPosterior(AbstractPosterior, ABC):
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        return self.reparametrized_sample(num_samples=num_samples).detach()

    def log_likelihood(self, samples: torch.Tensor) -> float:
        return self.reparametrized_sample_log_likelihoods(samples).detach()

    def trainable_parameters(self) -> List[Parameter]:
        return self.trainable_mean_parameters() + self.trainable_variance_parameters()

    @abstractmethod
    def trainable_mean_parameters(self) -> List[Parameter]:
        pass

    @abstractmethod
    def trainable_variance_parameters(self) -> List[Parameter]:
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

    @abstractmethod
    def reparametrized_sample_log_likelihoods(self, samples: torch.Tensor):
        pass
