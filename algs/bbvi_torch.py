"""
  bbvi.py (pytorch implementation)
  Black-box Variational Inference
  Author: Younhun Kim
"""

from abc import ABCMeta, abstractmethod
import torch

_CPU = torch.device("cpu")
_CUDA = torch.device("cuda")


# ============================ Constants =============================
default_device = _CUDA



class AbstractBBVIModel(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, model_params):
        super(AbstractBBVIModel, self).__init__()
        self.model_params = model_params

    @abstractmethod
    def rand_samples(self, params, num_samples):
        """
        Generate random samples from the restricted-family (e.g. Mean-Field) posterior distribution over R^d.
        :param params: Parametrization for the posterior.
        :param num_samples: N = Number of samples.
        :return: Outputs an N x d tensor of samples.
        """
        pass

    # Joint Distribution
    @abstractmethod
    def log_prob(self, z):
        """
        Compute the log-posterior of latent variables z.
        """
        pass

    def callback(self, *args):
        """
        Optional method called once per optimization step.
        """
        pass

    def compute_elbo(self, num_samples=1000):
        samples = self.rand_samples(self.model_params, num_samples)
        