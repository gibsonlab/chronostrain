"""
 generative.py
 Contains classes for representing the generative model.
"""

import numpy as np
from typing import List
from model.bacteria import Population
from model.reads import AbstractErrorModel, SequenceRead


def softmax(x: np.ndarray) -> np.ndarray:
    y = np.exp(x)
    return y / np.sum(y)


class GenerativeModel:
    def __init__(self,
                 times: List[int],
                 mu: np.ndarray,
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
        self.fragment_frequencies = self.bacteria_pop.get_strain_fragment_frequencies(window_size=read_length)

    def sample_abundances_and_reads(self, read_depths):
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

    def sample_timed_reads(self, abundances: List[np.ndarray], read_depths: List[int]):
        if len(abundances) != len(self.times):
            raise ValueError(
                "Length of strain_rel_abundances_motion ({}) must agree with number of time points ({})".format(
                    len(abundances), len(self.times)
                )
            )

        reads = []

        # For each time point, convert to fragment abundances and sample each read.
        for read_depth, strain_abundance in zip(read_depths, abundances):
            frag_abundance = self.strain_abundance_to_frag_abundance(strain_abundance)
            reads.append(self.sample_reads(frag_abundance, read_depth))

        return reads

    def time_scale(self, time_idx: int):
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
            return IndexError("Can't reference time at index {}.".format(time_idx))

    def _sample_brownian_motion(self) -> List[np.ndarray]:
        """
        Generates an S-dimensional brownian motion, S = # of strains, centered at mu.
        Initial covariance is tau_1, while subsequent variance scaling is tau.
        """
        brownian_motion = []
        covariance = np.identity(self.mu.size)
        center = self.mu  # Initialize mean vector.

        for time_idx in range(len(self.times)):
            scaling = self.time_scale(time_idx)
            center = np.random.multivariate_normal(center, covariance * scaling)
            brownian_motion.append(center)

        return brownian_motion

    def sample_abundances(self):
        abundances = []
        gaussians = self._sample_brownian_motion()
        for Z in gaussians:
            abundances.append(softmax(Z))
        return abundances

    def strain_abundance_to_frag_abundance(self, strain_abundance: np.ndarray) -> np.ndarray:
        """
        Convert strain abundance to fragment abundance, via the matrix multiplication F = WS.
        """
        return np.matmul(self.fragment_frequencies, strain_abundance)

    def sample_reads(self, frag_abundances: np.ndarray, num_samples: int = 1) -> List[SequenceRead]:
        """
        Given a set of fragments and their time indexed frequencies (based on the current time
        index strain abundances and the fragments' relative frequencies in each strain's sequence.),
        generate a set of noisy fragment reads where the read fragments are selected in proportion
        to their time indexed frequencies and the outputted base pair at location i of each selected
        fragment is chosen from a probability distribution condition on the actual base pair at location
        i and the quality score at location i in the generated quality score vector for the
        selected fragment.

        @param - time_indexed_fragment_frequencies -
                a list of floats representing a probability distribution over the fragments

        @param - fragments
                a list of strings representing the fragments

        @return - generated_noisy_fragments
                a list of strings representing a noisy reads of the set of input fragments

        """

        frag_space = self.bacteria_pop.get_fragment_space(self.read_length)
        samples = np.random.choice(list(frag_space.get_fragments()), num_samples, p=frag_abundances)

        # Draw a read from each fragment, do an in-place replacement.
        for i in range(num_samples):
            frag = samples[i]
            samples[i] = self.error_model.sample_noisy_read(frag.seq)

        return samples
