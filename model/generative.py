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
    def __init__(self, times, mu, tau_1, tau, W, strains, fragments, read_error_model):
        self.times = times
        self.mu = mu
        self.tau_1 = tau_1
        self.tau = tau
        self.W = W
        self.strains = strains
        self.fragments = fragments
        self.error_model = read_error_model

    def num_strains(self):
        """
        :return: The total number of strains in the model.
        """
        return len(self.strains)

    def fragments(self):
        """
        :return: A list (or an iterator) over the collection of unique fragments in the model.
        """

    def num_fragments(self):
        """
        :return: The total number of unique fragments in the model.
        """
        raise NotImplementedError()

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

