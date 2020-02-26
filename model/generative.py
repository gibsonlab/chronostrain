"""
 generative.py
 Contains classes for representing the generative model.
"""


class SequenceRead:
    """
    A class representing a sequence-quality vector pair.
    """
    def __init__(self, seq, quality, metadata):
        if len(seq) == len(quality):
            raise ValueError(
                "Sequence and Q-vector must match in length. Got: seq-len = {}, Q-len={}".format(len(seq), len(quality))
            )
        self.seq = seq
        self.quality = quality
        self.metadata = metadata

    def get_seq(self):
        return self.seq

    def get_quality(self):
        return self.quality


class GenerativeModel:
    def __init__(self, times, mu, tau_1, tau, W, strains, fragment_space, read_error_model):
        self.times = times  # array of time points
        self.mu = mu  # mean for X_1
        self.tau_1 = tau_1  # covariance scaling for X_1
        self.tau = tau  # covariance base scaling
        self.W = W  # the fragment-strain frequency matrix
        self.strains = strains  # array of strains
        self.fragment_space = fragment_space  # The set/enumeration (to be decided) of all possible fragments.
        self.error_model = read_error_model  # instance of (child class of) AbstractErrorModel

    def num_strains(self):
        """
        :return: The total number of strains in the model.
        """
        return len(self.strains)

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

    def sample_reads(self, num_samples):
        """
        Generate a time-indexed list of read collections.
        :param num_samples: the number of samples at each time point.
        :return:
        """
        if len(num_samples) != len(self.times):
            raise ValueError("Length of num_samples ({}) must agree with number of time points ({})".format(
                len(num_samples), len(self.times))
            )

        # TODO: implement this (call functions / copy-paste from scripts/model.py as necessary)
        # e.g. first generate brownian motion trajectory, then sample fragments, then sample reads.
        pass

    def sample_abundances(self):
        """

        :return: abundances AND reads (time-indexed)
        """

        # TODO implement this using code from scripts/model.py
        abundances = None
        reads = None

        return abundances, reads

