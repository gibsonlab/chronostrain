"""
    Black Box variational Inference with reprametrization trick
    Author: Sawal Acharya
"""
import os
from typing import List

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from chronostrain import cfg
from chronostrain.algs import AbstractModelSolver
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


class NaiveMeanFieldPosterior:
    def __init__(self,
                 model: GenerativeModel,
                 read_counts: List[int],
                 read_likelihoods: List[torch.tensor],
                 lr: float):

        self.model = model
        self.read_counts = read_counts
        self.reads_ll = read_likelihoods
        self.lr = lr

        self.W = self.model.get_fragment_frequencies()  # P(F= f | S = s); stochastic matrix whose column sum to 1
        self.times = self.model.times
        self.T = self.model.num_times()
        self.S = self.model.num_strains()

        # variational parameters (of the Gaussian posterior approximation)
        self.mu_all, self.sigma_all = self.initialize_vi_params()
        self.opt_mu = torch.optim.SGD(self.mu_all, lr=self.lr)
        self.opt_sigma = torch.optim.SGD(self.sigma_all, lr=self.lr)

        self.phi = {}
        self.elbo_all = []

    def initialize_vi_params(self):
        """initializes mu and sigma of the approximate posterior distribution,
           return lists whose length is equal to total expt. time"""

        mean_li = []
        std_li = []
        for t in range(0, self.T + 1):
            mean = torch.nn.Parameter(torch.rand(self.S))
            mean_li.append(mean)

            std = torch.tensor([2.], requires_grad=True, device=cfg.torch_cfg.device)
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
                    # print(phi_n.shape)
                    # print(torch.sum(phi_t[-1]))
                phi_all[i] = torch.stack(phi_t)
        # the keys must be the same as self.times
        # print(phi_all.keys() == self.times)
        return phi_all

    def compute_elbo(self, x_li) -> torch.Tensor:
        inst_elbo = torch.tensor([0.], device=cfg.torch_cfg.device)
        t_prev = torch.tensor([0.], device=cfg.torch_cfg.device)
        for i in range(1, self.T + 1):
            t = self.times[i - 1]
            v_scale = t - t_prev

            # TODO: @Sawal fix this to use self.model.tau_1 and self.model.tau separately.
            inst_elbo += -self.S * 0.5 * torch.log(2 * pi * (self.model.tau ** 2))
            inst_elbo += -0.5 / ((self.model.tau ** 2) * v_scale) * (
                    2 * torch.dot(x_li[i], x_li[i - 1])
                    + torch.dot(x_li[i], x_li[i])
                    + torch.dot(x_li[i - 1], x_li[i - 1])
            )
            inst_elbo += self.S / 2 * (torch.log(self.sigma_all[i] ** 2) + torch.log(2 * pi * e))
            t_prev = t

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
        total_elbo = torch.zeros(size=(1,), device=cfg.torch_cfg.device, dtype=cfg.torch_cfg.default_dtype)
        self.opt_mu.zero_grad()
        self.opt_sigma.zero_grad()
        q = MultivariateNormal(torch.zeros(self.S), torch.eye(self.S))
        for i in range(n_samples):
            for t in range(self.T + 1):
                samp = self.mu_all[t] + self.sigma_all[t] * q.sample()
                x_samples.append(samp)

            self.phi = self.update_phi(x_samples)
            total_elbo += self.compute_elbo(x_samples)
        total_elbo: torch.Tensor = total_elbo / n_samples
        loss: torch.Tensor = -total_elbo
        loss.backward()

        self.elbo_all.append(total_elbo)
        self.opt_mu.step()
        self.opt_sigma.step()

    def get_elbo(self) -> torch.Tensor:
        """
        Returns the elbo of the latest iteration
        """
        if len(self.elbo_all) != 0:
            return self.elbo_all[-1]
        else:
            raise KeyError("No ELBO value found; update_params() must be called at least once.")

    def get_mean(self):
        return self.mu_all

    def get_std(self):
        return self.sigma_all


class BBVIReparamSolver(AbstractModelSolver):
    """
    VI solver with naive mean-field assumption.
    """

    def __init__(self,
                 model: GenerativeModel,
                 data: List[List[SequenceRead]],
                 cache_tag: CacheTag,
                 out_base_dir: str,
                 read_likelihoods: List[torch.Tensor] = None
                 ):
        super().__init__(model, data, cache_tag, read_likelihoods=read_likelihoods)
        self.model = model
        self.read_counts = [len(reads_t) for reads_t in data]

        self.W = self.model.get_fragment_frequencies()
        self.times = self.model.times
        self.T = self.model.num_times()
        self.S = self.model.num_strains()

        self.out_base_dir = out_base_dir

    def solve(self,
              iters=100,
              thresh=1e-5,
              print_debug_every=200,
              lr=1e-3):
        posterior = NaiveMeanFieldPosterior(
            model=self.model,
            read_counts=self.read_counts,
            read_likelihoods=self.read_likelihoods,
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
        logger.info("Finished {it} iterations. Final ELBO: {elbo}".format(it=iters, elbo=elbo))

        # =========== Diagnostic values.
        final_mean = posterior.get_mean()
        final_std = posterior.get_std()

        mean_li = []
        std_li = []
        mean_soft = []

        for i in range(1, len(final_mean)):
            mean_soft.append(softmax(final_mean[i]).detach().numpy())
            mean_li.append(final_mean[i].detach().numpy())
            std_li.append(final_std[i].detach().numpy())

        np.savetxt(os.path.join(self.out_base_dir, "vi_abundance_bb_non_repar_2.csv"), np.asarray(mean_soft), delimiter=",")
        np.savetxt(os.path.join(self.out_base_dir, "bbvi_mean_non_repar_2.csv"), np.asarray(mean_li), delimiter=",")
        np.savetxt(os.path.join(self.out_base_dir, "bbvi_std_non_repar_2.csv"), np.asarray(std_li), delimiter=",")
        np.savetxt(os.path.join(self.out_base_dir, "bbvi_elbo_non_repar_2.csv"), np.asarray(posterior.elbo_all), delimiter=",")