"""
Monte-Carlo Variational Inference
(Variational Sequential Monte Carlo)
"""

import torch
from typing import List

from torch.distributions import MultivariateNormal, Categorical

from algs.vi import AbstractVariationalPosterior
from model.generative import GenerativeModel
from model.reads import SequenceRead
from algs.base import AbstractModelSolver, compute_read_likelihoods
from util.benchmarking import RuntimeEstimator

from util.io.logger import logger
from util.torch import multi_logit


class VariationalSequentialPosterior(AbstractVariationalPosterior):
    def __init__(
            self,
            num_times: int,
            num_strains: int,
            num_fragments: int,
            read_counts: List[int],
            device
    ):
        self.times = num_times
        self.strains = num_strains
        self.fragments = num_fragments
        self.read_counts = read_counts
        self.device = device

        # ================= Learnable parameters:
        # The mean parameters of the GP.
        # t = 0: the mean of the GP.
        # t > 0: Describes the conditional mean shift A, as in
        #     E[X_{t+1} | X_{t} = y] = E[X_{t+1}] + TRANSITION[t+1]*(y - E[X_{t}])
        #
        self.means = [
            torch.nn.Parameter(
                torch.zeros(num_strains-1, device=device, dtype=torch.double),
                requires_grad=True
            )
            for _ in range(num_times)
        ]

        # Represents the transition matrix Sigma_{t+1,t} * inv(Sigma_{t,t})
        # These describe the means of the conditional distribution X_{t+1} | X_{t}.
        self.transitions = [
            torch.nn.Parameter(
                torch.eye(num_strains-1, num_strains-1, device=device, dtype=torch.double),
                requires_grad=True
            )
            for _ in range(num_times - 1)
        ]

        # Represents the time-t covariances.
        # Describes the CONDITIONAL covariances Cov(X_{t+1} | X_{t}) -->
        # think of "Sigma_{i,j}" as the (time-i, time-j) block of the complete Covariance matrix.
        # t > 1: Sigma_{t+1,t+1} - Sigma_{t+1,t}*inv(Sigma_{t,t})*Sigma_{t,t+1}
        # t = 1: Sigma_{1,1}
        self.cond_covar_cholesky = [
            torch.nn.Parameter(
                torch.eye(num_strains-1, device=device, dtype=torch.double),
                requires_grad=True
            )
            for _ in range(num_times)
        ]

    def params(self) -> List[torch.nn.Parameter]:
        """
        Return a list of all learnable parameters (e.g. Parameter instances with requires_grad=True).
        """
        return self.means + self.transitions + self.cond_covar_cholesky
        # return self.means

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

    def sample(self, num_samples=1, apply_softmax=False) -> List[torch.Tensor]:
        # Indexing: (time) x (sample idx) x (distribution dimension)
        X = self.rand_sample_X(num_samples)
        if apply_softmax:
            X = [multi_logit(x_t, dim=1) for x_t in X]
        return X

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


class VSMCSolver(AbstractModelSolver):
    def __init__(self, model: GenerativeModel, data: List[List[SequenceRead]], torch_device):
        super().__init__(model, data)
        self.read_likelihoods = compute_read_likelihoods(model, data, logarithm=False, device=torch_device)
        self.posterior = VariationalSequentialPosterior(
            num_times=model.num_times(),
            num_strains=model.num_strains(),
            num_fragments=model.num_fragments(),
            read_counts=[len(reads) for reads in data],
            device=torch_device
        )
        self.device = torch_device

    def elbo_surrogate_estimate(self, num_samples) -> torch.Tensor:
        """
        Compute the ELBO surrogate approximation [Equation (9) of Naesseth et al
        (https://arxiv.org/pdf/1705.11140.pdf)].

        This implementation has been checked to be consistent with the authors' original implementation
        (https://github.com/blei-lab/variational-smc/blob/master/variational_smc.py).

        In the above style, this implementation uses the reparametrization trick and ignores the score derivative.
        (Auto-differentiation is only turned on for the weights to estimate p_hat(Y).
        Refer to Equation (9) of Naesseth et al (https://arxiv.org/pdf/1705.11140.pdf).

        :param num_samples:
        :return: A scalar tensor containing the ELBO surrogate value (with gradient information).
        """
        W_log_means = []
        W_normalized = torch.empty([1])
        X_prev = torch.empty([1])

        W_summands = []

        for t in range(self.posterior.times):
            # Propagation
            if t == 0:
                # Initial distribution of X_0 (no resampling required).
                distribution = MultivariateNormal(
                    loc=self.posterior.means[0].expand(num_samples, -1),
                    covariance_matrix=(self.posterior.cond_covar_cholesky[0]
                                       .t()
                                       .mm(self.posterior.cond_covar_cholesky[0]))
                )
            else:
                # Resampling (double-check that gradient is turned off.)
                children = Categorical(probs=W_normalized).sample([num_samples])  # Offsprings of each particle.

                # distribution of X_t.
                distribution = MultivariateNormal(
                    loc=((X_prev[children] - self.posterior.means[t-1].expand(num_samples, -1))
                         .mm(self.posterior.transitions[t-1])
                         + self.posterior.means[t].expand(num_samples, -1)),
                    covariance_matrix=(self.posterior.cond_covar_cholesky[t]
                                       .t()
                                       .mm(self.posterior.cond_covar_cholesky[t]))
                )
            X = distribution.sample().detach()

            # Compute weights.
            logW = self.model.log_likelihood_xt(
                t=t, X=X, X_prev=X_prev, read_likelihoods=self.read_likelihoods[t]
            ) - distribution.log_prob(X)

            # Resampling probabilities.
            W_normalized = (logW - logW.max()).detach().exp()  # Turn off gradients for re-sampling.
            W_normalized = W_normalized / W_normalized.sum()
            X_prev = X

            W_summands.append((W_normalized * logW).sum())
        return torch.stack(W_summands).sum()

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

        logger.debug("VSMC algorithm started. (Target iterations = {it})".format(
            it=iters
        ))
        time_est = RuntimeEstimator(total_iters=iters, horizon=print_debug_every)
        for i in range(1, iters+1, 1):
            time_est.stopwatch_click()

            # Maximize elbo by minimizing -elbo.
            elbo = -self.elbo_surrogate_estimate(num_samples=num_samples)
            optimizer.zero_grad()
            elbo.backward()

            optimizer.step()

            millis_elapsed = time_est.stopwatch_click()
            time_est.increment(millis_elapsed)
            if i % print_debug_every == 0:
                logger.info("Iteration {i} | time left: {t:.2f} min. | Last ELBO = {elbo}".format(
                    i=i,
                    t=time_est.time_left() / 60000,
                    elbo=elbo.item()
                ))
