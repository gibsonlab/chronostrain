"""
 generative.py
 Contains classes for representing the generative model.
"""
import sys
from typing import List, Tuple

import torch
from tqdm import tqdm
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import softmax

from chronostrain.config import cfg
from chronostrain.model.bacteria import Population
from chronostrain.model.fragments import FragmentSpace
from chronostrain.model.reads import AbstractErrorModel, SequenceRead
from . import logger


class GenerativeModel:
    def __init__(self,
                 times: List[float],
                 mu: torch.Tensor,
                 tau_1: float,
                 tau: float,
                 bacteria_pop: Population,
                 read_error_model: AbstractErrorModel,
                 read_length: int):

        self.times = times  # array of time points
        self.mu = mu  # mean for X_1
        self.tau_1 = tau_1  # covariance scaling for X_1
        self.tau = tau  # covariance base scaling
        self.error_model = read_error_model
        self.bacteria_pop = bacteria_pop
        self.read_length = read_length

    def num_times(self) -> int:
        return len(self.times)

    def num_strains(self) -> int:
        return len(self.bacteria_pop.strains)

    def num_fragments(self) -> int:
        return self.get_fragment_space().size()

    def get_fragment_space(self) -> FragmentSpace:
        return self.bacteria_pop.get_fragment_space(self.read_length)

    def get_fragment_frequencies(self) -> torch.Tensor:
        """
        Outputs the (F x S) matrix representing the strain-specific fragment frequencies.
        Is a wrapper for Population.get_strain_fragment_frequencies().
        (Corresponds to the matrix "W" in writeup.)
        """
        return self.bacteria_pop.get_strain_fragment_frequencies(window_size=self.read_length)

    def log_likelihood_x(self,
                         X: List[torch.Tensor],
                         read_likelihoods: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the joint log-likelihood of X, F, and R according to this generative model.
        Let N be the number of samples.
        :param X: The S-dimensional Gaussian trajectory, indexed (T x N x S) as a List of 2-d tensors.
        :param read_likelihoods: A precomputed list of tensors containing read-fragment log likelihoods
          (output of compute_read_likelihoods(logarithm=False).)
        :return: The log-likelihood from the generative model. Outputs a length-N
        tensor (one log-likelihood for each sample).
        """
        ans = torch.zeros(size=[X[0].size(0)], device=X[0].device)
        for t, X_t in enumerate(X):
            ans = ans + self.log_likelihood_xt(
                t=0,
                X=X_t,
                X_prev=X[t-1] if t > 0 else None,
                read_likelihoods=read_likelihoods[t]
            )
        return ans

    def log_likelihood_xt(self,
                          t: int,
                          X: torch.Tensor,
                          X_prev: torch.Tensor,
                          read_likelihoods: torch.Tensor):
        """
        :param t: the time index (0 thru T-1)
        :param X: (N x S) tensor of time (t) samples.
        :param X_prev: (N x S) tensor of time (t-1) samples. (None if t=0).
        :param read_likelihoods: An (F x R_t) matrix of conditional read probabilities.
        :return: The joint log-likelihood p(X_t, Reads_t | X_{t-1}).
        """
        # Gaussian part
        N = X.size(0)
        if t == 0:
            center = self.mu.repeat(N, 1)
        else:
            center = X_prev
        covariance = self.time_scale(t) * torch.eye(self.num_strains(), device=cfg.torch_cfg.device)
        gaussian_log_probs = MultivariateNormal(loc=center, covariance_matrix=covariance).log_prob(X)

        # Reads likelihood calculation, conditioned on the Gaussian part.
        data_log_probs = (softmax(X, dim=1)
                          .mm(self.get_fragment_frequencies().t())
                          .mm(read_likelihoods)
                          .log()
                          .sum(dim=1))

        return gaussian_log_probs + data_log_probs

    def sample_abundances_and_reads(
            self,
            read_depths: List[int]
    ) -> Tuple[torch.Tensor, List[List[SequenceRead]]]:
        """
        Generate a time-indexed list of read collections and strain abundances.

         :param read_depths: the number of reads per time point. A time-indexed array of ints.
         :return: a tuple of (abundances, reads) where
             strain_abundances is
                 a time-indexed list of 1D numpy arrays of fragment abundances based on the
                 corresponding time index strain abundances and the fragments' relative
                 frequencies in each strain's sequence.
             reads is
                a time-indexed list of lists of SequenceRead objects, where the i-th
                inner list corresponds to a reads taken at time index i (self.times[i])
                and contains num_samples[i] read objects.
        """
        if len(read_depths) != len(self.times):
            raise ValueError("Length of num_samples ({}) must agree with number of time points ({})".format(
                len(read_depths), len(self.times))
            )

        abundances = self.sample_abundances()
        reads = self.sample_timed_reads(abundances, read_depths)
        return abundances, reads

    def sample_timed_reads(self, abundances: torch.Tensor, read_depths: List[int]) -> List[List[SequenceRead]]:
        S = self.num_strains()
        F = self.num_fragments()
        num_timepoints = len(read_depths)

        # Invoke call to force intialization.
        self.bacteria_pop.get_strain_fragment_frequencies(self.read_length)

        logger.debug("Sampling reads, conditioned on abundances.")

        if abundances.size(0) != len(self.times):
            raise ValueError(
                "Argument abundances (len {}) must must have specified number of time points (len {})".format(
                    abundances.size(0), len(self.times)
                )
            )

        reads_list = []

        # For each time point, convert to fragment abundances and sample each read.
        for t in tqdm(range(num_timepoints), file=sys.stdout):
            read_depth = read_depths[t]
            strain_abundance = abundances[t]
            frag_abundance = self.strain_abundance_to_frag_abundance(strain_abundance.view(S, 1)).view(F)
            reads_list.append(self.sample_reads(frag_abundance, read_depth, metadata="SIM_t{}".format(self.times[t])))

        return reads_list

    def time_scale(self, time_idx: int) -> float:
        """
        Return the k-th time increment.
        :param time_idx: the index to query (corresponding to k).
        :return: the time differential t_k - t_(k-1).
        """

        if time_idx == 0:
            return self.tau_1
        if time_idx < len(self.times):
            return self.tau * (self.times[time_idx] - self.times[time_idx-1])
        else:
            raise IndexError("Can't reference time at index {}.".format(time_idx))

    def _sample_brownian_motion(self) -> torch.Tensor:
        """
        Generates an S-dimensional brownian motion, S = # of strains, centered at mu.
        Initial covariance is tau_1, while subsequent variance scaling is tau.
        """
        brownian_motion = []
        covariance = torch.eye(list(self.mu.size())[0], device=cfg.torch_cfg.device)  # Initialize covariance matrix
        center = self.mu  # Initialize mean vector.

        for time_idx in range(len(self.times)):
            scaling = self.time_scale(time_idx)
            center = MultivariateNormal(loc=center, covariance_matrix=(scaling ** 2) * covariance).sample()
            brownian_motion.append(center)

        return torch.stack(brownian_motion, dim=0)

    def sample_abundances(self) -> torch.Tensor:
        """
        Samples abundances from a Gaussian Process.

        :return: A T x S tensor; each row is an abundance profile for a time point.
        """
        gaussians = self._sample_brownian_motion()
        return softmax(gaussians, dim=1)

    def strain_abundance_to_frag_abundance(self, strain_abundances: torch.Tensor) -> torch.Tensor:
        """
        Convert strain abundance to fragment abundance, via the matrix multiplication F = WZ.
        Assumes strain_abundance is an S x T tensor, so that the output is an F x T tensor.
        """
        return self.get_fragment_frequencies().mm(strain_abundances)

    def sample_reads(
            self,
            frag_abundances: torch.Tensor,
            num_samples: int = 1,
            metadata: str = "") -> List[SequenceRead]:
        """
        Given a set of fragments and their time indexed frequencies (based on the current time
        index strain abundances and the fragments' relative frequencies in each strain's sequence.),
        generate a set of noisy fragment reads where the read fragments are selected in proportion
        to their time indexed frequencies and the outputted base pair at location i of each selected
        fragment is chosen from a probability distribution condition on the actual base pair at location
        i and the quality score at location i in the generated quality score vector for the
        selected fragment.

        @param - frag_abundances -
                a tensor of floats representing a probability distribution over the fragments

        @param - num_samples -
                the number of samples to be taken.

        @return - generated_noisy_fragments
                a list of strings representing a noisy reads of the set of input fragments

        """

        frag_indexed_samples = torch.multinomial(
            frag_abundances,
            num_samples=num_samples,
            replacement=True
        )

        frag_samples = []
        frag_space = self.bacteria_pop.get_fragment_space(self.read_length)

        # Draw a read from each fragment.
        for i in range(num_samples):
            frag = frag_space.get_fragment_by_index(frag_indexed_samples[i].item())
            frag_samples.append(self.error_model.sample_noisy_read(frag, metadata=(metadata + "|" + frag.metadata)))

        return frag_samples
