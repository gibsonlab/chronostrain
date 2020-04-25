"""
 generative.py
 Contains classes for representing the generative model.
"""

import torch

from typing import List, Tuple
from model.bacteria import Population
from model.fragments import FragmentSpace
from model.reads import AbstractErrorModel, SequenceRead
from util.io.logger import logger

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.nn.functional import softmax


class GenerativeModel:
    def __init__(self,
                 times: List[int],
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
        """
        return self.bacteria_pop.get_strain_fragment_frequencies(window_size=self.read_length)

    def log_likelihood_torch(
            self,
            X: List[torch.Tensor],
            F: List[torch.Tensor],
            R: List[List[SequenceRead]],
            device) -> torch.Tensor:
        """
        Computes the joint log-likelihood of X, F, and R.
        Let N be the number of samples.
        :param X: The S-dimensional Gaussian trajectory, indexed (T x N x S) as a List of 2-d tensors.
        :param F: The per-read (R reads) sampled fragments, indexed (T x N x R_t) as a list of 2-d tensors.
        :param R: The sampled reads, indexed (T x R_t) as a list of list of SequenceReads.
        :param device: The torch device to run the calculations on.
        :return: The log-likelihood from the generative model. Outputs a length-N
        tensor (one log-likelihood for each sample).
        """
        n = X[0].size(0)
        ans = torch.zeros(n, dtype=torch.double, device=device)
        prev_x = torch.tensor(len(self.mu), device=device)
        for t in range(len(X)):
            # ==== Note:
            # each x_t is an N x S matrix.
            x_t = X[t]
            dist = MultivariateNormal(
                loc=prev_x.mm,
                covariance_matrix=self.time_scale(t) * torch.eye(self.num_strains())
            )
            ans = ans + dist.log_prob(x_t)
            prev_x = x_t

            # ==== Note:
            # y_t is an N x S matrix.
            # W is a F x S matrix, want to end up with an N x F matrix.
            # Proper dimension ordering is y_t * transpose(W).
            y_t = softmax(x_t, dim=1)
            frag_freqs = self.get_fragment_frequencies()  # the W matrix
            z_t = y_t.mm(frag_freqs.transpose(0, 1))  # an N x F matrix, each row is a frag frequency vector.

            # ==== Note:
            # each f_t is an N x R_t matrix.
            f_t = F[t]
            ans = ans + Categorical(z_t).log_prob(f_t)
            for m in range(n):
                for i, read in enumerate(R[t]):
                    ans[m] = ans[m] + self.error_model.compute_log_likelihood(
                        fragment=self.get_fragment_space().get_fragment_by_index(i),
                        read=read
                    )
        return ans

    def sample_abundances_and_reads(
            self,
            read_depths: List[int]
    ) -> Tuple[List[torch.Tensor], List[List[SequenceRead]]]:
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

    def sample_timed_reads(self, abundances: List[torch.Tensor], read_depths: List[int]) -> List[List[SequenceRead]]:
        S = self.num_strains()
        F = self.num_fragments()
        logger.debug("Sampling reads, conditioned on abundances.")

        if len(abundances) != len(self.times):
            raise ValueError(
                "Argument abundances (len {}) must must have specified number of time points (len {})".format(
                    len(abundances), len(self.times)
                )
            )

        reads_list = []

        # For each time point, convert to fragment abundances and sample each read.
        for k in range(len(read_depths)):
            read_depth = read_depths[k]
            strain_abundance = abundances[k]
            frag_abundance = self.strain_abundance_to_frag_abundance(strain_abundance.view(S, 1)).view(F)
            reads_list.append(self.sample_reads(frag_abundance, read_depth, metadata="Simulated read t={}".format(self.times[k])))

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

    def _sample_brownian_motion(self) -> List[torch.Tensor]:
        """
        Generates an S-dimensional brownian motion, S = # of strains, centered at mu.
        Initial covariance is tau_1, while subsequent variance scaling is tau.
        """
        brownian_motion = []
        covariance = torch.eye(list(self.mu.size())[0])  # Initialize covariance matrix
        center = self.mu  # Initialize mean vector.

        for time_idx in range(len(self.times)):
            scaling = self.time_scale(time_idx)
            mvn = MultivariateNormal(loc=center, covariance_matrix=scaling*covariance)
            center = mvn.sample()
            brownian_motion.append(center)

        return brownian_motion

    def sample_abundances(self) -> List[torch.Tensor]:

        logger.info("Generating abundances trajectory")
        abundances = []
        gaussians = self._sample_brownian_motion()
        for X in gaussians:
            abundances.append(softmax(X))
        return abundances

    def strain_abundance_to_frag_abundance(self, strain_abundance: torch.Tensor) -> torch.Tensor:
        """
        Convert strain abundance to fragment abundance, via the matrix multiplication F = WZ.
        Assumes strain_abundance is an S x T tensor, so that the output is an F x T tensor.
        """
        return self.get_fragment_frequencies().mm(strain_abundance)

    def sample_reads(self, frag_abundances: torch.Tensor, num_samples: int = 1, metadata: str = "") -> List[SequenceRead]:
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
            frag_samples.append(self.error_model.sample_noisy_read(frag.seq, metadata=metadata))

        return frag_samples
