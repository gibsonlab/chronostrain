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

from algs.base import AbstractModelSolver
from model.generative import GenerativeModel
from util.io.logger import logger
from util.benchmarking import RuntimeEstimator


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
        # The initial mean of the GP.
        self.mean = torch.nn.Parameter(torch.zeros(
            strains, device=device, dtype=torch.double
        ), requires_grad=True)

        # Represents the transition matrix Sigma_{t+1,t} * inv(Sigma_{t,t}).
        self.transitions = [torch.nn.Parameter(torch.eye(
            strains, strains, device=device, dtype=torch.double
        ), requires_grad=True) for _ in range(times)]

        # Represents the precision matrices inv(Sigma_{t+1,t+1} - Sigma_{t+1,t} inv(Sigma_{t,t}) * Sigma_{t,t+1}).
        self.precisions = [torch.nn.Parameter(torch.eye(
            strains, device=device, dtype=torch.double
        ), requires_grad=True) for _ in range(times)]

        # The categorical weights for each time point.
        self.frag_weights = [torch.nn.Parameter(torch.ones(
            fragments, device=device, dtype=torch.double
        ), requires_grad=True) for _ in range(times)]

    def params(self) -> List[torch.nn.Parameter]:
        """
        Return a list of all learnable parameters (e.g. Parameter instances with requires_grad=True).
        """
        return [self.mean] + self.transitions + self.precisions + self.frag_weights

    def sample(self, num_samples=1) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Indexing: (time) x (sample idx) x (distribution dimension)
        X = self.rand_sample_X(num_samples)
        F = self.rand_sample_F(num_samples)
        return X, F

    def log_likelihood(self, X, F) -> torch.Tensor:
        return self.log_likelihood_X(X) + self.log_likelihood_F(F)

    def rand_sample_X(self, num_samples) -> List[torch.Tensor]:
        # Dimension indexing: (T x N x S)
        X = [torch.empty(1) for _ in range(self.times)]
        prev_x = self.mean.view(self.strains).repeat(num_samples, 1)
        for t in range(self.times):
            X[t] = MultivariateNormal(
                loc=prev_x.mm(self.transitions[t]),
                precision_matrix=self.precisions[t]
            ).sample().to(self.device)
            X[t].requires_grad = False
            prev_x = X[t]
        #
        # # Dimension indexing: (N x T x S)
        # X = torch.stack(X).transpose(0,1)
        return X

    def rand_sample_F(self, num_samples) -> List[torch.Tensor]:
        # Indexing: (T x N x R)
        F = [
            Categorical(self.frag_weights[t])
                .sample([num_samples, self.read_counts[t]])
                .to(self.device)
            for t in range(self.times)
        ]
        for t in range(self.times):
            F[t].requires_grad = False
        #
        # # Indexing: (N x T x R)
        # F = torch.stack(F).transpose(0,1)
        return F

    def log_likelihood_X(self, X: List[torch.Tensor]) -> torch.Tensor:
        N = X[0].size(0)
        ans = torch.zeros(N, dtype=torch.double, device=self.device)
        prev_x = self.mean.view(1, self.strains)
        for t in range(self.times):
            dist = MultivariateNormal(loc=prev_x.mm(self.transitions[t]), precision_matrix=self.precisions[t])
            ans = ans + dist.log_prob(X[t])
            prev_x = X[t]
        return ans

    def log_likelihood_F(self, F: List[torch.Tensor]) -> torch.Tensor:
        N = F[0].size(0)
        ans = torch.zeros(N, dtype=torch.double, device=self.device)
        for t in range(self.times):
            dist = Categorical(self.frag_weights[t])
            ans = ans + dist.log_prob(F[t]).sum(dim=1)
        return ans


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

    def elbo_estimate(self, num_samples=1000):
        """
        Computes the monte-carlo approximation to the ELBO objective, as specified in
        https://arxiv.org/abs/1401.0118.
        (No re-parametrization trick, no Rao-Blackwell-ization.)
        """
        (X_samples, F_samples) = self.posterior.sample(num_samples=num_samples)

        posterior_ll = self.posterior.log_likelihood(
            X=X_samples,
            F=F_samples
        )
        posterior_ll_grad_disabled = posterior_ll.clone()
        posterior_ll_grad_disabled.requires_grad = False

        generative_ll = self.model.log_likelihood_torch(
            X=X_samples,
            F=F_samples,
            R=self.data,
            device=self.device
        )

        elbo_monte_carlo = posterior_ll * (generative_ll - posterior_ll_grad_disabled)
        return elbo_monte_carlo.mean()

    def solve(self,
              optim_class=torch.optim.SGD,
              optim_args=None,
              iters=4000,
              num_samples=1000,
              print_debug_every=200
              ):
        if optim_args is None:
            optim_args = {'lr': 1e-4}
        optimizer = optim_class(
            self.posterior.params(),
            **optim_args
        )

        time_est = RuntimeEstimator(total_iters=iters, horizon=10)
        for i in range(1, iters+1, 1):
            time_est.stopwatch_click()

            loss = -self.elbo_estimate(num_samples=num_samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            if i % print_debug_every == 0:
                logger.debug("Iteration {i} | time left: {t} min. | Last ELBO = {loss}".format(
                    i=i,
                    t=time_est.time_left() // 60,
                    loss=loss.item()
                ))
