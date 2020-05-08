"""
  bbvi.py (pytorch implementation)
  Black-box Variational Inference
  Author: Younhun Kim
"""

from typing import List, Tuple
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical

from model.reads import SequenceRead

from algs.base import AbstractModelSolver, compute_read_likelihoods
from model.generative import GenerativeModel
from util.io.logger import logger
from util.benchmarking import RuntimeEstimator
from util.torch import multi_logit

class MeanFieldPosterior:
    def __init__(
            self,
            times: int,
            strains: int,
            fragments: int,
            read_counts: List[int],
            device
    ):
        """
        Mean-field assumption:
        1) Parametrize X_1, ..., X_T as a Gaussian Process, with covariance kernel \Sigma_{t, t+1}.
        2) Parametrize F_1, ..., F_T as independent (but not identical) categorical RVs (for each read).
        :param times: Number of time points, T.
        :param strains: Number of strains, S.
        :param fragments: Number of fragments, F.
        :param read_counts: Number of reads per time point.
        :param device: the device to use for pytorch. (Default: CUDA if available).
        """
        # Check: might need this to be a matrix, not a vector.
        self.times = times
        self.strains = strains
        self.fragments = fragments
        self.read_counts = read_counts
        self.device = device

        # ================= Learnable parameters:
        # The mean parameters of the GP.
        # t = 0: the mean of the GP.
        # t > 0: Describes the conditional mean shift A, as in
        #     E[X_{t+1} | X_{t} = y] = E[X_{t+1}] + TRANSITION[t+1]*(y - E[X_{t}])
        #                            = TRANSITION[t+1]*y - A
        self.means = [
            torch.nn.Parameter(
                torch.zeros(strains - 1, device=device, dtype=torch.double),
                requires_grad=True
            )
            for _ in range(times)
        ]

        # Represents the transition matrix Sigma_{t+1,t} * inv(Sigma_{t,t})
        # These describe the means of the conditional distribution X_{t+1} | X_{t}.
        self.transitions = [
            torch.nn.Parameter(
                torch.eye(strains - 1, strains - 1, device=device, dtype=torch.double),
                requires_grad=True
            )
            for _ in range(times-1)
        ]

        # Represents the time-t covariances.
        # Describes the CONDITIONAL covariances Cov(X_{t+1} | X_{t}) -->
        # think of "Sigma_{i,j}" as the (time-i, time-j) block of the complete Covariance matrix.
        # t > 1: Sigma_{t+1,t+1} - Sigma_{t+1,t}*inv(Sigma_{t,t})*Sigma_{t,t+1}
        # t = 1: Sigma_{1,1}
        self.cond_covar_cholesky = [
            torch.nn.Parameter(
                torch.eye(strains - 1, device=device, dtype=torch.double),
                requires_grad=True
            )
            for _ in range(times)
        ]

    def params(self) -> List[torch.nn.Parameter]:
        """
        Return a list of all learnable parameters (e.g. Parameter instances with requires_grad=True).
        """
        return self.means + self.transitions + self.cond_covar_cholesky

    def sample(self, num_samples=1, apply_softmax=False) -> List[torch.Tensor]:
        # Indexing: (time) x (sample idx) x (distribution dimension)
        X = self.rand_sample_X(num_samples)
        if apply_softmax:
            X = [multi_logit(x_t, dim=1) for x_t in X]
        return X

    def log_likelihood(self, X) -> torch.Tensor:
        return self.log_likelihood_X(X)

    def rand_sample_X(self, num_samples) -> List[torch.Tensor]:
        # Dimension indexing: (T instances of N x S tensors)
        X = [torch.empty(1) for _ in range(self.times)]
        for t in range(self.times):
            if t == 0:
                X[0] = MultivariateNormal(
                    loc=self.means[0].expand(num_samples, -1),
                    covariance_matrix=self.cond_covar_cholesky[0].t().mm(self.cond_covar_cholesky[0])
                ).sample().to(self.device)
            else:
                X[t] = MultivariateNormal(
                    loc=((X[t-1] - self.means[t-1].expand(num_samples, -1))
                         .mm(self.transitions[t-1])
                         + self.means[t].expand(num_samples, -1)),
                    covariance_matrix=self.cond_covar_cholesky[t].t().mm(self.cond_covar_cholesky[t])
                ).sample().to(self.device)
            X[t].requires_grad = False
        return X

    def log_likelihood_X(self, X: List[torch.Tensor]) -> torch.Tensor:
        N = X[0].size(0)
        ans = torch.zeros(N, dtype=torch.double, device=self.device)

        for t in range(self.times):
            if t == 0:
                dist = MultivariateNormal(
                    loc=self.means[0].expand(N, -1),
                    covariance_matrix=self.cond_covar_cholesky[0].t().mm(self.cond_covar_cholesky[0])
                )
            else:
                dist = MultivariateNormal(
                    loc=((X[t-1] - self.means[t-1].expand(N, -1))
                         .mm(self.transitions[t-1])
                         + self.means[t].expand(N, -1)),
                    covariance_matrix=self.cond_covar_cholesky[t].t().mm(self.cond_covar_cholesky[t])
                )
            ans = ans + dist.log_prob(X[t])
        return ans

    def get_means_detached(self):
        return [mean.detach() for mean in self.means]

    def get_variances_detached(self):
        variances = []
        prev_covariance = None
        for t in range(self.times):
            if t == 0:
                covariance = self.cond_covar_cholesky[0].t().mm(self.cond_covar_cholesky[0])
            else:
                covariance = self.cond_covar_cholesky[t].t().mm(self.cond_covar_cholesky[t]) \
                             + self.transitions[t-1].mm(prev_covariance.mm(self.transitions[t-1].t()))
            variances.append(covariance.diagonal().detach())
            prev_covariance = covariance
        return variances


class BBVISolver(AbstractModelSolver):
    """
    An abstraction of a black-box VI implementation.
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: List[List[SequenceRead]],
                 device):
        super(BBVISolver, self).__init__(model, data)
        self.model = model
        self.data = data
        self.device = device
        self.posterior = MeanFieldPosterior(
            times=model.num_times(),
            strains=model.num_strains(),
            fragments=model.num_fragments(),
            read_counts=[len(reads) for reads in data],
            device=device
        )
        self.read_likelihoods = compute_read_likelihoods(self.model, self.data, logarithm=False, device=device)

    def elbo_estimate(self, num_samples=1000):
        """
        Computes the monte-carlo approximation to the ELBO objective, as specified in
        https://arxiv.org/abs/1401.0118.
        (No re-parametrization trick, no Rao-Blackwell-ization.)
        """
        X_samples = self.posterior.sample(num_samples=num_samples)
        posterior_ll = self.posterior.log_likelihood(X_samples)
        posterior_ll_grad_disabled = posterior_ll.detach()
        generative_ll = self.model.log_likelihood_x(X=X_samples, read_likelihoods=self.read_likelihoods)
        elbo_samples = posterior_ll * (generative_ll - posterior_ll_grad_disabled)
        return elbo_samples.mean()

    def solve(self,
              optim_class=torch.optim.Adam,
              optim_args=None,
              iters=4000,
              num_samples=8000,
              print_debug_every=200):
        if optim_args is None:
            optim_args = {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.}
        optimizer = optim_class(
            self.posterior.params(),
            **optim_args
        )

        logger.debug("BBVI algorithm started. (Gradient method, Target iterations={it})".format(
            it=iters
        ))
        time_est = RuntimeEstimator(total_iters=iters, horizon=print_debug_every)
        for i in range(1, iters+1, 1):
            time_est.stopwatch_click()

            elbo_loss = -self.elbo_estimate(num_samples=num_samples)
            optimizer.zero_grad()
            elbo_loss.backward()
            optimizer.step()

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            if i % print_debug_every == 0:
                gradient_size = 0.
                for param in self.posterior.params():
                    gradient_size += param.grad.norm(p=2).item()

                logger.info("Iteration {i} | time left: {t:.2f} min. | Last ELBO = {elbo} | Gradient norm = {grad}".format(
                    i=i,
                    t=time_est.time_left() / 60000,
                    elbo=-elbo_loss.item(),
                    grad=gradient_size
                ))
