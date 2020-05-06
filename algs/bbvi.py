"""
  bbvi.py (pytorch implementation)
  Black-box Variational Inference
  Author: Younhun Kim
"""

from typing import List, Tuple
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.nn.functional import softmax

from model.reads import SequenceRead

from algs.base import AbstractModelSolver, compute_read_likelihoods
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
        # The mean parameters of the GP.
        # t = 0: the mean of the GP.
        # t > 0: Describes the conditional mean shift A, as in
        #     E[X_{t+1} | X_{t} = y] = E[X_{t+1}] + TRANSITION[t+1]*(y - E[X_{t}])
        #                            = TRANSITION[t+1]*y - A
        self.means = [
            torch.nn.Parameter(
                torch.zeros(strains, device=device, dtype=torch.double),
                requires_grad=True
            )
            for _ in range(times)
        ]

        # Represents the transition matrix Sigma_{t+1,t} * inv(Sigma_{t,t})
        # These describe the means of the conditional distribution X_{t+1} | X_{t}.
        self.transitions = [
            torch.nn.Parameter(
                torch.eye(strains, strains, device=device, dtype=torch.double),
                requires_grad=True
            )
            for _ in range(times-1)
        ]

        # Represents the time-t covariances.
        # Describes the conditional covariances Cov(X_{t+1} | X_{t}) -->
        # think of "Sigma_{i,j}" as the (time-i, time-j) block of the complete Covariance matrix.
        # t > 1: Sigma_{t+1,t+1} - Sigma_{t+1,t}*inv(Sigma_{t,t})*Sigma_{t,t+1}
        # t = 1: Sigma_{1,1}
        self.covariances = [
            torch.nn.Parameter(
                torch.eye(strains, device=device, dtype=torch.double),
                requires_grad=True
            )
            for _ in range(times)
        ]

        # The categorical weights for each time point.
        self.frag_weights = [
            torch.nn.Parameter(
                torch.ones(size=[read_counts[t], fragments], device=device, dtype=torch.double),
                requires_grad=True
            ) for t in range(times)
        ]

    def params(self) -> List[torch.nn.Parameter]:
        """
        Return a list of all learnable parameters (e.g. Parameter instances with requires_grad=True).
        """
        return self.means + self.transitions + self.covariances + self.frag_weights

    def sample(self, num_samples=1, apply_softmax=False) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Indexing: (time) x (sample idx) x (distribution dimension)
        X = self.rand_sample_X(num_samples)
        if apply_softmax:
            X = [softmax(x_t, dim=1) for x_t in X]
        F = self.rand_sample_F(num_samples)
        return X, F

    def log_likelihood(self, X, F) -> torch.Tensor:
        return self.log_likelihood_X(X) + self.log_likelihood_F(F)

    def rand_sample_X(self, num_samples) -> List[torch.Tensor]:
        # Dimension indexing: (T instances of N x S tensors)
        X = [torch.empty(1) for _ in range(self.times)]
        prev_x = None
        for t in range(self.times):
            if t == 0:
                X[0] = MultivariateNormal(
                    loc=self.means[0].repeat(num_samples, 1),
                    covariance_matrix=self.covariances[0]
                ).sample().to(self.device)
            else:
                X[t] = MultivariateNormal(
                    loc=prev_x.mm(self.transitions[t]) - self.means[t].repeat(num_samples, 1),
                    covariance_matrix=self.covariances[t]
                ).sample().to(self.device)
            X[t].requires_grad = False
            prev_x = X[t]
        return X

    def rand_sample_F(self, num_samples) -> List[torch.Tensor]:
        # Indexing: (T instances of N x R tensors)
        F = [
            (Categorical(self.frag_weights[t])
             .sample([num_samples])
             .to(self.device))
            for t in range(self.times)
        ]
        for t in range(self.times):
            F[t].requires_grad = False
        return F

    def log_likelihood_X(self, X: List[torch.Tensor]) -> torch.Tensor:
        N = X[0].size(0)
        ans = torch.zeros(N, dtype=torch.double, device=self.device)

        print("COVARIANCE MATRIX:")
        print(self.covariances[0])
        print("HERE!")

        logger.info("Posterior sample:")
        sample_x, sample_f = self.sample(num_samples=1, apply_softmax=True)
        logger.info("X: {}".format(sample_x))
        logger.info("F: {}".format(sample_f))

        print("**********************")



        for t in range(self.times):
            if t == 0:
                dist = MultivariateNormal(
                    loc=self.means[0].repeat(N, 1),
                    covariance_matrix=self.covariances[0]
                )
            else:
                dist = MultivariateNormal(
                    loc=X[t-1].mm(self.transitions[t]) - self.means[t].repeat(N, 1),
                    covariance_matrix=self.covariances[t]
                )
            ans = ans + dist.log_prob(X[t])
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
        self.read_log_likelihoods = compute_read_likelihoods(self.model, self.data, logarithm=True, device=device)

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
        posterior_ll_grad_disabled = posterior_ll.detach()

        generative_ll = self.model.log_likelihood_torch(
            X=X_samples,
            F=F_samples,
            read_log_likelihoods=self.read_log_likelihoods,
            device=self.device
        )

        elbo_samples = posterior_ll * (generative_ll - posterior_ll_grad_disabled)
        return elbo_samples.mean()

    def solve(self,
              optim_class=torch.optim.SGD,
              optim_args=None,
              iters=4000,
              num_samples=1000,
              print_debug_every=200):

        if optim_args is None:
            optim_args = {'lr': 1e-7}
        optimizer = optim_class(
            self.posterior.params(),
            **optim_args
        )

        logger.debug("BBVI algorithm started. (Gradient method, Target iterations={})".format(
            iters
        ))
        time_est = RuntimeEstimator(total_iters=iters, horizon=5)
        for i in range(1, iters+1, 1):
            time_est.stopwatch_click()

            loss = -self.elbo_estimate(num_samples=num_samples)
            print("ELBO loss = {}".format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)

            if i % print_debug_every == 0:
                logger.debug("Iteration {i} | time left: {t} min. | Last ELBO = {loss}".format(
                    i=i,
                    t=time_est.time_left() // 60,
                    loss=-loss.item()
                ))
