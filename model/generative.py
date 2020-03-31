"""
 generative.py
 Contains classes for representing the generative model.
"""

import numpy as np


class SequenceRead:
    """
    A class representing a sequence-quality vector pair.
    """

    def __init__(self, seq, quality, metadata):

        if len(seq) != len(quality):
            raise ValueError("Length of nucleotide sequence ({}) must agree with length "
                             "of quality score sequence ({})".format(len(seq), len(quality))
                             )

        if type(seq) != str:
            raise ValueError("seq must be of type string")

        if type(quality) != np.ndarray:
            raise ValueError("quality must be a numpy array")

        for q_score in quality:
            if type(q_score) != np.int64 or type(q_score) == int:
                raise ValueError("Each value in the quality vector must be numpy ints or ints")

        self.seq = seq  # string
        self.quality = quality  # np array of ints
        self.metadata = metadata


class GenerativeModel:

    def __init__(self, times, mu, tau_1, tau, W, fragment_space, read_error_model, bacteria_pop=None):

        self.times = times  # array of time points
        self.mu = mu  # mean for X_1
        self.tau_1 = tau_1  # covariance scaling for X_1
        self.tau = tau  # covariance base scaling
        self.W = W  # the fragment-strain frequency matrix
        self.fragment_space = fragment_space  # The set/enumeration (to be decided) of all possible fragments.
        self.error_model = read_error_model  # instance of (child class of) AbstractErrorModel

        self.bacteria_pop = bacteria_pop
        self.strain_fragment_matrix = self.bacteria_pop.generate_strain_fragment_frequencies(self.fragment_space)

    def num_strains(self):
        """
        :return: The total number of strains in the model.
        """

        return len(self.bacteria_pop.strains)

    def sample_reads(self, num_samples):
        """
        A wrapper function for sample_abundances_and_reads
        Generate a time-indexed list of read collections.

         :param num_samples: A list of the number of samples to take each time point.
            (e.g. num_samples[i] is the number of samples we take at time point self.times[i])

         :return: a time-indexed list of lists of SequenceRead objects, where the i-th
                inner list corresponds to reads taken at time index i (self.times[i])
                and contains num_samples[i] read objects.
        """

        abundances, reads = self.sample_abundances_and_reads(num_samples)
        return reads

    def sample_abundances(self, num_samples):
        """
         A wrapper function for sample_abundances_and_reads

         See sample_abundances_and_reads for details.
         """

        abundances, reads = self.sample_abundances_and_reads(num_samples)
        return abundances

    def sample_abundances_and_reads(self, num_samples):
        """
        Generate a time-indexed list of read collections and strain abundances.

         :param num_samples: A list of the number of samples to take each time point.
            (e.g. num_samples[i] is the number of samples we take at time point self.times[i])

         :return: a tuple of (abundances, reads) where
             abundances is
                 a time-indexed list of 1D numpy arrays of fragment abundances based on the
                 corresponding time index strain abundances and the fragments' relative
                 frequencies in each strain's sequence.
             reads is
                a time-indexed list of lists of SequenceRead objects, where the i-th
                inner list corresponds to a reads taken at time index i (self.times[i])
                and contains num_samples[i] read objects.
        """

        if len(num_samples) != len(self.times):
            raise ValueError("Length of num_samples ({}) must agree with number of time points ({})".format(
                len(num_samples), len(self.times))
            )

        # Step 1
        brownian_motion = self.generate_brownian_motion()

        # Step 2
        strain_rel_abundances_motion = self.generate_relative_abundances(brownian_motion)

        if len(strain_rel_abundances_motion) != len(self.times):
            raise ValueError("Length of strain_rel_abundances_motion ({}) must agree "
                             "with number of time points ({})".format(
                                len(strain_rel_abundances_motion), len(self.times)))

        reads = []
        abundances = []

        # Iterate over each time step.
        # For each time step, we have a particular number of reads we want to sample
        # and the bacteria strains have a particular set of relative abundances.
        for num_sample, strain_rel_abnd in zip(num_samples, strain_rel_abundances_motion):
            # Step 3
            time_indexed_fragment_frequencies = self.generate_time_indexed_fragment_frequencies(strain_rel_abnd)
            abundances.append(time_indexed_fragment_frequencies)

            # Step 4
            time_indexed_reads = self.generate_reads(time_indexed_fragment_frequencies, num_sample)
            reads.append(time_indexed_reads)

        return abundances, reads

    def time_scale(self, time_idx):
        """
        Return the k-th time increment.
        :param time_idx: the index to query (corresponding to k).
        :return: the time differential t_k - t_(k-1).
        """

        if time_idx == 0:
            return self.tau_1
        if time_idx < len(self.times):
            return self.tau * (self.times[time_idx] - self.times[time_idx] - 1)
        else:
            return IndexError("Can't reference time at index {}.".format(time_idx))

    # Step 1
    def generate_brownian_motion(self):
        """
        Generates multivariate gaussian motion.
        There is one dimension for each strain in the multivariate gaussian.
        Variance is applied according to tau_1, tau, and the time differences present in
        the

        The number of dimenions at each time point is equal to the number of strains
        and the number of time points is equ


        @return --
            a list of numpy arrays, where each array has length equal to the number of strains.
        """

        brownian_motion = []

        covar_matrix = np.identity(self.mu.size)
        mean_vec = self.mu  # Initialize mean vector.

        for time_idx in range(len(self.times)):
            time_scale_value = self.time_scale(time_idx)
            mean_vec = np.random.multivariate_normal(mean_vec, covar_matrix * (time_scale_value ** 2))
            brownian_motion.append(mean_vec)

        return np.asanyarray(brownian_motion)

    # Step 2
    def generate_relative_abundances(self, gaussian_process):
        """
        @param - gaussain_process - an list of n-dimensional arrays of floats

        @return - rel_abundances
            a list of n-dimension arrays after running softmax on each one.
        """

        rel_abundances_motion = []

        for sample in gaussian_process:
            # softmax
            total = sum(np.exp(sample))
            rel_abundances = [np.exp(x) / total for x in sample]

            rel_abundances_motion.append(rel_abundances)

        if not all([round(sum(x), 4) == 1 for x in rel_abundances_motion]):
            ValueError("Sum of relative strain abundances at each time point should sum to 1")

        return rel_abundances_motion

    # Step 3
    def generate_time_indexed_fragment_frequencies(self, strain_abundances):
        """
        @param - strain_fragment_matrix -
            A 2D numpy array where column i is the relative frequencies of observing each
            of the fragments in strain i.

        @param - strain_abundances
            A list representing the relative abundances of the strains

        @returns - fragment_to_prob_vector
             a 1D numpy array of time indexed fragment abundances based on the current time index strain abundances
             and the fragments' relative frequencies in each strain's sequence.
        """

        fragment_to_prob_vector = np.matmul(self.strain_fragment_matrix, np.array(strain_abundances))

        if round(sum(fragment_to_prob_vector), 4) != 1:
            raise ValueError("Relative fragment probabilities (for a given time point) should sum to 1")

        return fragment_to_prob_vector

    # Step 4
    def generate_reads(self, time_indexed_fragment_frequencies, num_reads=None):
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

        if not num_reads:
            num_reads = len(self.fragment_space)

        # Sample fragments in proportion to their time indexed frequencies.
        sampled_fragments = np.random.choice(self.fragment_space, num_reads, p=time_indexed_fragment_frequencies)

        generated_noisy_fragments = []

        # For each sampled fragment, generate a noisy read of it.
        for frag in sampled_fragments:
            generated_noisy_fragments.append(self.error_model.sample_noisy_read(frag))

        return generated_noisy_fragments

# TODO: Write reads and quality scores to file (low priority)
