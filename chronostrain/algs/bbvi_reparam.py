"""
    Black Box variational Inference with reprametrization trick
    Author: Sawal Acharya
"""
from typing import List

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from chronostrain import cfg
from chronostrain.algs import AbstractModelSolver, AbstractPosterior
from chronostrain.model import *
from chronostrain.model.reads import *
from chronostrain.util.data_cache import CacheTag
from . import logger


# === Constants
pi = torch.tensor([np.pi], device=cfg.torch_cfg.device)
e = torch.tensor([np.exp(1)], device=cfg.torch_cfg.device)


def softmax(x: torch.Tensor):
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x)


class NaiveMeanFieldPosterior(AbstractPosterior):
    def __init__(self,
                 model: GenerativeModel,
                 read_counts: List[int],
                 read_log_likelihoods: List[torch.tensor],
                 lr: float):
        # Model parameters.
        self.model = model
        self.W = self.model.get_fragment_frequencies()  # P(F= f | S = s); stochastic matrix whose column sum to 1
        self.times = self.model.times
        self.T = self.model.num_times()
        self.S = self.model.num_strains()

        # Data likelihoods.
        self.reads_ll = read_log_likelihoods
        self.read_counts = read_counts

        # variational parameters (of the Gaussian posterior approximation) and their optimizers.
        self.lr = lr
        self.mu_parameters, self.sigma_parameters = self.initialize_vi_params()
        self.opt_mu = torch.optim.SGD(self.mu_parameters, lr=self.lr)
        self.opt_sigma = torch.optim.SGD(self.sigma_parameters, lr=self.lr)

        self.phi = {}
        self.elbo_history = []

    def initialize_vi_params(self):
        """
        Initializes mu and sigma of the approximate posterior distribution, return lists whose
        length is equal to total expt. time
        """

        mean_li = []
        std_li = []
        for t in range(0, self.T + 1):
            if t == 0:
                mean = self.model.mu
                std = torch.tensor(self.model.tau_1)
            else:
                mean = torch.nn.Parameter(torch.rand(self.S))
                std = torch.tensor([2.], requires_grad=True, device=cfg.torch_cfg.device)
            mean_li.append(mean)
            std_li.append(std)

        return mean_li, std_li

    def update_phi(self, x_li):
        """updates the probabilities of fragment assignments
           returns a dictionary whose keys are the time where reads are sampled"""

        phi_all = {}
        with torch.no_grad():
            for i in range(1, self.T + 1):
                reads_t = self.reads_ll[i - 1]
                phi_t = []
                x_soft = softmax(x_li[i])
                w_t = torch.matmul(self.W, x_soft)

                for n_t in range(self.read_counts[i - 1]):
                    phi_n = torch.exp(torch.log(w_t) + reads_t[:, n_t])
                    # debugging tool : must sum to 1 since this is a probability

                    phi_t.append(phi_n / torch.sum(phi_n))
                    phi_t.append(phi_n)

                    # print(phi_n.shape)
                    # print(torch.sum(phi_t[-1]))
                phi_all[i] = torch.stack(phi_t)

        # the keys must be the same as self.times
        # print(phi_all.keys() == self.times)
        return phi_all

    def compute_elbo_lower_bound(self, x_li) -> torch.Tensor:
        inst_elbo = torch.tensor([0.], device=cfg.torch_cfg.device)

        # The initial value of x (t_idx = 0)
        inst_elbo += -self.S * 0.5 * torch.log(2 * pi * (self.model.tau_1 ** 2))
        inst_elbo += -0.5 * (1 / (self.model.tau_1 ** 2)) * torch.pow(x_li[0] - self.model.mu, 2)

        t_prev = 0
        for i in range(self.model.num_times()):
            inst_elbo += -0.5 * (1 / self.model.time_scaled_variance(i)) * torch.sum(torch.pow(x_li[i] - x_li[i - 1], 2))
            inst_elbo += self.S / 2 * (torch.log(self.sigma_parameters[i] ** 2) + torch.log(2 * pi * e))

        for i in range(1, self.T + 1):
            x_soft = softmax(x_li[i])
            w_t = torch.matmul(self.W, x_soft)
            for n in range(self.read_counts[i - 1]):
                prob = self.phi[i][n]
                t_n = torch.log(w_t) + self.reads_ll[i - 1][:, n]
                inst_elbo += torch.sum(t_n * prob)

        return inst_elbo

    def update_params(self):
        """
        Updates the parameters of approximate posterior.
        """
        n_samples = 10
        x_samples = []
        total_elbo = torch.zeros(size=(1,), device=cfg.torch_cfg.device)
        self.opt_mu.zero_grad()
        self.opt_sigma.zero_grad()
        q = MultivariateNormal(torch.zeros(self.S), torch.eye(self.S))

        for i in range(n_samples):
            for t in range(self.T + 1):
                samp = self.mu_parameters[t] + self.sigma_parameters[t] * q.sample()
                x_samples.append(samp)

            self.phi = self.update_phi(x_samples)
            total_elbo += self.compute_elbo_lower_bound(x_samples)
        total_elbo = total_elbo / n_samples
        loss = -total_elbo
        loss.backward()

        self.elbo_history.append(total_elbo)
        self.opt_mu.step()
        self.opt_sigma.step()

    def get_elbo(self) -> torch.Tensor:
        """
        Returns the elbo of the latest iteration
        """
        if len(self.elbo_history) != 0:
            return self.elbo_history[-1]
        else:
            raise KeyError("No ELBO value found; update_params() must be called at least once.")

    def get_mean(self):
        return self.mu_parameters

    def get_std(self):
        return self.sigma_parameters

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        :param num_samples:
        :return: A (T x N x S) tensor of time-indexed abundance samples.
        """
        time_indexed_samples = []
        for t_idx in range(1, self.T + 1):
            mean_t = self.mu_parameters[t_idx].detach()
            sigma_t = self.sigma_parameters[t_idx].detach()

            posterior_t = torch.distributions.MultivariateNormal(
                loc=mean_t,
                covariance_matrix=torch.pow(sigma_t, 2) * torch.eye(self.S, device=cfg.torch_cfg.device)
            )

            samples = posterior_t.sample(sample_shape=(num_samples,))
            time_indexed_samples.append(samples)
        return torch.stack(time_indexed_samples)


class BBVIReparamSolver(AbstractModelSolver):
    """
    VI solver with naive mean-field assumption.
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: List[List[SequenceRead]],
                 cache_tag: CacheTag,
                 out_base_dir: str
                 ):
        super().__init__(model, data, cache_tag)
        self.read_counts = [len(reads_t) for reads_t in data]
        self.W = self.model.get_fragment_frequencies()
        self.times = self.model.times
        self.T = self.model.num_times()
        self.S = self.model.num_strains()
        self.out_base_dir = out_base_dir

    def solve(self,
              iters=100,
              thresh=1e-5,
              print_debug_every=10,
              lr=1e-3):
        posterior = NaiveMeanFieldPosterior(
            model=self.model,
            read_counts=self.read_counts,
            read_log_likelihoods=self.read_log_likelihoods,
            lr=lr
        )
        logger.debug("Black Box Variational Inference Algorithm (with reparametrization) started. "
                     "Target iterations={it}, thresh={thresh})".format(it=iters, thresh=thresh))

        for i in range(1, iters + 1):
            posterior.update_params()
            if i % print_debug_every == 0:
                logger.debug("Iteration # {}, ELBO: {}".format(i, posterior.get_elbo().item()))
                mean = posterior.get_mean()
                std_dev = posterior.get_std()
                for t in range(1, self.T + 1):
                    logger.debug("time: {}, mean: {}, rel_abund: {}, std: {}".format(
                        t,
                        mean[t].detach().numpy(),
                        softmax(mean[t]).detach().numpy(),
                        std_dev[t].detach().numpy()
                    ))
        logger.info("Finished {it} iterations. Final ELBO: {elbo}".format(
            it=iters, elbo=posterior.get_elbo()
        ))

        return posterior
