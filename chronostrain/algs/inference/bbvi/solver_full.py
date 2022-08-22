from pathlib import Path
from typing import List, Optional, Callable, Type, Dict, Any, Tuple

import math
import numba
from numba import njit

import numpy as np
import scipy
import scipy.special, scipy.stats
import torch
from torch.distributions import Normal

from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads

from chronostrain.config import cfg
from chronostrain.util.math.psis import psis_smooth_ratios

from .. import AbstractModelSolver
from .util import log_softmax_t
from chronostrain.util.math.matrices import log_mm_exp
from .solver_gaussian import ADVIGaussianSolver
from ..vi import AbstractPosterior

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class ADVISolverFullPosterior(AbstractModelSolver):

    def __init__(self,
                 model: GenerativeModel,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 read_batch_size: int = 5000,
                 num_cores: int = 1,
                 partial_correlation_type: str = 'strain'):
        logger.info("Initializing full correlation solver (ADVI + Importance weighted estimation)")
        AbstractModelSolver.__init__(
            self, model, data, db, num_cores=num_cores
        )
        self.partial_solver = ADVIGaussianSolver(
            model, data, db, read_batch_size, num_cores,
            correlation_type=partial_correlation_type
        )
        self.posterior: GaussianPosteriorFullCorrelation = None
        self.log_smoothed_weights: torch.Tensor = None
        self.k_hat: float = float('inf')

    def prior_ll(self, x_samples: torch.Tensor) -> torch.Tensor:
        return self.model.log_likelihood_x(X=x_samples).detach()

    def data_ll(self, x_samples: torch.Tensor) -> torch.Tensor:
        ans = torch.zeros(size=(x_samples.shape[1],), device=x_samples.device)
        for t_idx in range(self.model.num_times()):
            log_y_t = log_softmax_t(x_samples, t=t_idx)
            for batch_lls in self.partial_solver.batches[t_idx]:
                batch_matrix = log_mm_exp(log_y_t, batch_lls).detach()
                ans = ans + torch.sum(batch_matrix, dim=1)
        return ans

    def solve(self,
              optimizer_class: Type[torch.optim.Optimizer],
              optimizer_args: Dict[str, Any],
              temp_dir: Path,
              num_epochs: int = 1,
              iters: int = 4000,
              num_bbvi_samples: int = 500,
              num_importance_samples: int = 5000,
              batch_size: int = 1000,
              min_lr: float = 1e-4,
              lr_decay_factor: float = 0.25,
              lr_patience: int = 10,
              callbacks: Optional[List[Callable[[int, torch.Tensor, float], None]]] = None):
        """
        :param optimizer_class:
        :param optimizer_args:
        :param temp_dir: A directory to save temporary files to (for batch sampling).
        :param num_epochs:
        :param iters:
        :param num_bbvi_samples:
        :param num_importance_samples: The number of samples to use for importance sampling to estimate full posterior.
        :param min_lr:
        :param lr_decay_factor:
        :param lr_patience:
        :param callbacks:
        :return:
        """
        """First, solve using the mean-field factorized posteriors."""
        self.partial_solver.solve(
            optimizer_class, optimizer_args,
            num_epochs, iters, num_bbvi_samples,
            min_lr, lr_decay_factor, lr_patience,
            callbacks
        )

        """Next, estimate the true posterior covariance."""
        logger.debug("Performing importance-weighted estimation of parameters from learned posterior.")

        temp_dir.mkdir(exist_ok=True, parents=True)
        log_importance_weights = []
        num_batches = int(np.ceil(num_importance_samples / batch_size))

        from tqdm import tqdm
        for batch_idx in tqdm(range(num_batches), desc="Batched sampling"):
            batch_start_idx = batch_idx * batch_size
            this_batch_sz = min(num_importance_samples - batch_start_idx, batch_size)
            batch_weights = self.sample_batch(this_batch_sz, batch_idx, temp_dir)
            log_importance_weights.append(batch_weights)

        # normalize (we don't know normalization constant of true posterior).
        log_importance_weights = np.concatenate(log_importance_weights)
        log_importance_weights = log_importance_weights - scipy.special.logsumexp(log_importance_weights)
        log_smoothed_weights, k_hat = psis_smooth_ratios(log_importance_weights)

        logger.debug(f"Estimated Pareto k-hat: {k_hat}")
        if k_hat > 0.7:
            # Extremely large number of samples are needed for stable gradient estimates!
            logger.warning("Pareto k-hat estimate exceeds safe threshold (0.7). "
                           "Estimates may be biased/overfit to the data.")

        logger.debug("Computing importance-weighted mean and covariances.")
        mean, cov = self.estimate_mean_and_covar(temp_dir, num_batches, log_smoothed_weights)
        self.posterior = GaussianPosteriorFullCorrelation(
            self.model.num_strains(), self.model.num_times(),
            mean, cov,
            torch_device=cfg.torch_cfg.device
        )
        self.log_smoothed_weights = log_smoothed_weights
        self.k_hat = k_hat
        logger.debug("Finished computing full posterior estimate.")

    def estimate_mean_and_covar(self, batch_dir: Path, num_batches: int, log_importance_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gaussian_dim = self.model.num_times() * self.model.num_strains()
        mean_estimate = np.zeros(shape=gaussian_dim, dtype=np.float)
        cov_estimate = np.zeros(shape=(gaussian_dim, gaussian_dim), dtype=np.float)

        batch_start_idx = 0
        from tqdm import tqdm
        for batch_idx in tqdm(range(num_batches), desc="Importance sampling estimator"):
            # Load the batch
            samples = np.load(str(self.get_batch_path(batch_dir, batch_idx)))  # (N_batch) x (TS)
            batch_sz = samples.shape[0]
            batch_log_weights = log_importance_weights[batch_start_idx:batch_start_idx + batch_sz]

            mean_estimate += weighted_mean(samples, batch_log_weights)
            cov_estimate += weighted_cov(samples, batch_log_weights, eps=1e-6)

            # Prepare for next iteration
            batch_start_idx += batch_sz
        return mean_estimate, cov_estimate

    @staticmethod
    def get_batch_path(batch_dir: Path, batch_idx: int) -> Path:
        return batch_dir / f'FULL_COVAR_samples_batch_{batch_idx}.npy'

    def sample_batch(self, batch_size: int, batch_idx: int, batch_dir: Path) -> np.ndarray:
        # (T x N x S)
        x_samples = self.partial_solver.posterior.sample(num_samples=batch_size)

        # length N
        approx_posterior_ll = self.partial_solver.posterior.log_likelihood(x_samples)
        forward_ll = self.prior_ll(x_samples).detach() + self.data_ll(x_samples).detach()
        log_importance_ratios = forward_ll - approx_posterior_ll

        np.save(
            str(self.get_batch_path(batch_dir, batch_idx)),
            torch.transpose(x_samples, 0, 1).reshape(batch_size, self.model.num_times() * self.model.num_strains()).cpu().numpy()
        )
        return log_importance_ratios.cpu().numpy()


def weighted_mean(x: np.ndarray, log_w: np.ndarray) -> np.ndarray:
    """
    :param x: (N x D) 2-dimensional array.
    :param log_w: length-N array of log-weights.
    :return:
    """
    return np.sum(x * np.expand_dims(np.exp(log_w), axis=1), axis=0)


@njit(parallel=False)
def weighted_cov(x: np.ndarray, log_w: np.ndarray, eps: float) -> np.ndarray:
    """
    :param x: (N x D) 2-dimensional array.
    :param log_w: length-N array of log-weights.
    :return:
    """
    # Compute the column-wise mean
    x_mean = []
    for c in range(x.shape[1]):
        x_mean.append(x[:, c].mean())
    x_mean = np.array(x_mean)

    estimate = np.zeros((x.shape[1], x.shape[1]), dtype=numba.float64)
    for n in range(x.shape[0]):
        deviation = x[n, :] - x_mean  # n-th sample deviation X_n - X_mean
        weight = math.exp(log_w[n])
        for i in range(estimate.shape[0]):
            for j in range(estimate.shape[1]):
                estimate[i, j] += weight * deviation[i] * deviation[j]
    return estimate


class GaussianPosteriorFullCorrelation(AbstractPosterior):
    """
    A basic implementation, which specifies a bias vector `b` and linear transformation `A`.
    Samples using the reparametrization x = Az + b, where z is a vector of standard Gaussians, so that x has covariance
    (A.T)(A) and mean b.
    """

    def __init__(self, num_strains: int, num_times: int, bias: np.ndarray, cov: np.ndarray, torch_device):
        if len(bias) != num_strains * num_times:
            raise ValueError(
                "Expected `mean` argument to be of length {} ({} strains x {} timepoints), but got {}.".format(
                    num_strains * num_times,
                    num_strains,
                    num_times,
                    len(bias)
                )
            )
        self.num_strains = num_strains
        self.num_times = num_times
        self.torch_device = torch_device

        self.bias = torch.tensor(bias, device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
        eigvals, eigvecs = scipy.linalg.eigh(cov)

        neg_locs = np.where(eigvals < 0)[0]
        nonneg_locs = np.where(eigvals >= 0)[0]
        if len(neg_locs) > 0:
            logger.debug("Negative eigenvalues: {} out of {}".format(
                len(neg_locs), len(eigvals)
            ))
            eigvals = eigvals[nonneg_locs]
            eigvecs = eigvecs[:, nonneg_locs]

        self.rank = len(eigvals)
        self.linear = torch.tensor(
            eigvecs * np.expand_dims(np.sqrt(eigvals), axis=0),
            device=cfg.torch_cfg.device,
            dtype=cfg.torch_cfg.default_dtype
        )  # (TS x k)
        self.standard_normal = Normal(
            loc=torch.tensor(0.0, device=cfg.torch_cfg.device),
            scale=torch.tensor(1.0, device=cfg.torch_cfg.device)
        )

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        samples = self.standard_normal.sample(sample_shape=(num_samples, self.rank))  # N x k
        samples = samples @ torch.transpose(self.linear, 0, 1) + self.bias  # (N x k) @ (k x TS) + (TS)

        # Re-shape N x (TS) into (T x N x S).
        return torch.softmax(torch.transpose(
            samples.reshape(num_samples, self.num_times, self.num_strains),
            0, 1
        ), dim=2)

    def mean(self) -> torch.Tensor:
        return torch.Tensor(self.bias, device=cfg.torch_cfg.device)

    def log_likelihood(self, x: torch.Tensor) -> float:
        raise ValueError("Not implemented for Full posterior (due to numerical instability with singular covariances)")

    def save(self, target_path: Path):
        torch.save(
            {
                'bias': self.bias,
                'linear': self.linear
            },
            target_path
        )
