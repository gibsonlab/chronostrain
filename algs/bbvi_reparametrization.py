#Black Box variational Inference with reprametrization trick 
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
from torch.distributions import MultivariateNormal, Categorical

from model.generative import GenerativeModel
from model.reads import SequenceRead
from algs.base import AbstractModelSolver, compute_read_likelihoods

from util.torch import multi_logit
from util.benchmarking import RuntimeEstimator
from util.io.logger import logger

import numpy as np
import matplotlib.pyplot as plt
import copy

# constants
pi = torch.tensor([np.pi], dtype=torch.float)
e = torch.tensor([np.exp(1)], dtype=torch.float)

class NaiveMeanFieldPosterior():

    def __init__(self, model: GenerativeModel, read_counts: List[int],
                 read_likelihoods: List[torch.tensor], device):

        self.model = model
        self.read_counts = read_counts
        self.device = device
        self.reads_ll = read_likelihoods

        self.W = self.model.get_fragment_frequencies()  # P(F= f | S = s); stochastic matrix whose column sum to 1
        self.tau = torch.tensor([1], dtype=torch.float)  # std dev of Brownian motion
        self.times = self.model.times
        self.T = self.model.num_times()
        self.S = self.model.num_strains()

        print("Preliminaries")
        print("times:", self.times, "total_time:", self.T, "num strains:", self.S)
        print("read_counts:", self.read_counts, "reads length:", len(self.reads_ll))
        print()

        self.eta = 0.001
        # variational parameters (of the Gaussian posterior approximation)
        self.mu_all, self.sigma_all = self.initialize_vi_params()
        self.opt_mu = torch.optim.SGD(self.mu_all, lr = self.eta)
        self.opt_sigma = torch.optim.SGD(self.sigma_all, lr=self.eta)

        self.phi = {}
        self.elbo_all = []

    def softmax(self, x):

        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x)

    def initialize_vi_params(self):
        """initializes mu and sigma of the approximate posterior distribution,
           return lists whose length is equal to total expt. time"""

        mean_li = []
        std_li = []
        #len of mean_li must be len(self.times) + 1 (including data at t = 0)
        #mean_li = [torch.tensor([0.5, -0.05, -0.5], dtype = torch.float, requires_grad = True),
        ##torch.tensor([0.7, -0.01, -1], dtype = torch.float, requires_grad = True),
        #torch.tensor([1.09, 0.00677, -0.07], dtype = torch.float, requires_grad = True),
        #torch.tensor([1.01, 0.1, -1.1], dtype = torch.float, requires_grad = True),
        #torch.tensor([0.3, 0.7, -1], dtype = torch.float, requires_grad = True),
        #torch.tensor([0.4, 1.25, -1], dtype = torch.float, requires_grad = True),
        ##torch.tensor([0.7, 0.3, -1], dtype = torch.float, requires_grad = True),
        #torch.tensor([1.3, -0.3, -0.5], dtype = torch.float, requires_grad = True),
        #torch.tensor([2.0, -0.5, -0.25], dtype = torch.float, requires_grad = True)]
        for t in range(0, self.T + 1):
            mean = torch.nn.Parameter(torch.rand(self.S))
            mean_li.append(mean)

            std = torch.tensor([2], dtype = torch.float, requires_grad = True)
            std_li.append(std)
        #print(mean_li)
        #print(std_li)
        return mean_li, std_li

    def update_phi(self, x_li):
        """updates the probabilities of fragment assignments
           returns a dictionary whose keys are the time where reads are sampled"""

        phi_all = {}
        with torch.no_grad():
            for i in range(1, self.T + 1):
                reads_t = self.reads_ll[i - 1]
                t = self.times[i - 1]  # time at which the data was sampled
                phi_t = []
                x_soft = self.softmax(x_li[i])
                W_t = torch.matmul(self.W, x_soft)

                for n_t in range(self.read_counts[i - 1]):
                    phi_n = torch.exp(torch.log(W_t) + reads_t[:, n_t])
                    # debugging tool : must sum to 1 since this is a probability
                    phi_t.append(phi_n / torch.sum(phi_n))
                    #print(phi_n.shape)
                    #print(torch.sum(phi_t[-1]))
                phi_all[i] = torch.stack(phi_t)
        # the keys must be the same as self.times
        # print(phi_all.keys() == self.times)
        return phi_all

    def compute_elbo(self, x_li):

        score = 0
        inst_elbo = 0
        t_prev = 0
        for i in range(1, self.T + 1):
            t = self.times[i - 1]
            v_scale = t - t_prev
            #print(v_scale)

            inst_elbo += -self.S / 2 * (torch.log(self.tau ** 2) + torch.log(2 * pi))

            inst_elbo += -0.5 / ((self.tau ** 2) * v_scale) * (2 * torch.dot(x_li[i],
            x_li[i -1]) + torch.dot(x_li[i], x_li[i]) + torch.dot(x_li[i - 1],
            x_li[i - 1]))

            inst_elbo += self.S / 2 * (torch.log(self.sigma_all[i] ** 2) + torch.log(
                2 * pi * e))

            t_prev = t

        for i in range(1, self.T + 1):
            x_soft = self.softmax(x_li[i])
            t = self.times[i - 1]
            W_t = torch.matmul(self.W, x_soft)
            for n in range(self.read_counts[i -1]):
                prob = self.phi[i][n]
                t_n = torch.log(W_t) + self.reads_ll[i - 1][:, n]
                inst_elbo += torch.sum(t_n * prob)

        return inst_elbo

    def update_params(self):
        '''updates the parameters of approximate posterior'''

        n_samples = 10
        prob_dists = []
        x_samples = []
        total_elbo = 0
        self.opt_mu.zero_grad()
        self.opt_sigma.zero_grad()
        q = MultivariateNormal(torch.zeros(self.S), torch.eye(self.S))
        q_li = []
        #for t in range(self.T + 1):
         #   q = MultivariateNormal(self.mu_all[t], self.sigma_all[t]  * torch.eye(self.S))
          #  q_li.append(q)
        for i in range(n_samples):
            for t in range(self.T + 1):
                samp = self.mu_all[t]  + self.sigma_all[t] * q.sample()
                #q = MultivariateNormal(self.mu_all[t], (self.sigma_all[t] **2 )  * torch.eye(self.S))
                x_samples.append(samp)

            self.phi = self.update_phi(x_samples)
            total_elbo += self.compute_elbo(x_samples)
        total_elbo = total_elbo / n_samples
        loss = -total_elbo
        loss.backward()
        self.elbo_all.append(total_elbo)
        self.opt_mu.step()
        self.opt_sigma.step()

    def get_elbo(self):
        '''returns the elbo of the latest iteration'''

        if len(self.elbo_all) != 0:
            return self.elbo_all[-1]
        else:
            return None

    def get_mean(self):

        return self.mu_all

    def get_std(self):

        return self.sigma_all

class BBVISolver():
    '''VI solver with naive mean-field assumption, decouple markov chains '''

    def __init__(self,
                 model: GenerativeModel,
                 data: List[List[SequenceRead]],
                 torch_device):
        self.model = model
        self.read_counts = [len(reads) for reads in data]

        self.device = torch_device
        self.reads_ll = compute_read_likelihoods(model=model, reads=data,
                                                 logarithm=True, device=torch_device)  # read likelihoods in youn's code
        self.tau = torch.tensor([1], dtype=torch.float)
        self.W = self.model.get_fragment_frequencies()
        self.times = self.model.times
        self.T = self.model.num_times()
        self.S = self.model.num_strains()

    def softmax(self, x):

        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x)


    def run_vi(self, iters=100, thresh=1e-5):
        posterior = NaiveMeanFieldPosterior(model=self.model, read_counts=
        self.read_counts, read_likelihoods= self.reads_ll, device=self.device)
        # logger.debug("Varitional Inference Algorithm Started (Naive Mean Field).Target iterations={it})".format(
        # it=iters"))
        logger.debug("Black Box Variational Inference Algorithm")
        for i in range(1, iters + 1):
            posterior.update_params()
            print("Iteration #", i, "ELBO:", posterior.get_elbo().item())
            mean = posterior.get_mean()
            std_dev = posterior.get_std()
            for t in range(1, self.T + 1):
                print("time:", t, "mean:", mean[t], "rel_abundance:", self.softmax(mean[t]),
                      "std:", std_dev[t])
            print()
        final_mean = posterior.get_mean()
        final_std = posterior.get_std()

        mean_li = []
        std_li = []
        mean_soft = []

        for i in range(1, len(final_mean)):
            mean_soft.append(self.softmax(final_mean[i]).detach().numpy())
            mean_li.append(final_mean[i].detach().numpy())
            std_li.append(final_std[i].detach().numpy())

        np.savetxt("vi_abundance_bb_non_repar_2.csv", np.asarray(mean_soft), delimiter = ",")
        np.savetxt("bbvi_mean_non_repar_2.csv", np.asarray(mean_li), delimiter = ",")
        np.savetxt("bbvi_std_non_repar_2.csv", np.asarray(std_li), delimiter = ",")
        np.savetxt("bbvi_elbo_non_repar_2.csv", np.asarray(posterior.elbo_all),
                    delimiter = ",")
