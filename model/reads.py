"""
 reads.py
 Contains classes for the error model of reads.
"""
import math
from abc import ABCMeta, abstractmethod
import torch
from model.fragments import Fragment


# ============= Utility functions
def mutate_acgt(base):
    i = torch.randint(low=0, high=3, size=[1]).item()
    bases = {'A', 'C', 'G', 'T'}
    bases.remove(base)
    return list(bases)[i]

# ============= END Utility functions


class SequenceRead:
    """
    A class representing a sequence-quality vector pair.
    """
    def __init__(self, seq: str, quality: torch.Tensor, metadata: str):
        if len(seq) != len(quality):
            raise ValueError(
                "Length of nucleotide sequence ({}) must agree with length of quality score sequence ({})".format(
                    len(seq), len(quality)
                )
            )
        self.seq = seq
        self.quality = quality
        self.metadata = metadata


class AbstractQScoreDistribution(metaclass=ABCMeta):
    """
    Parent class for all Q-score distributions.
    """

    def __init__(self, length):
        self.length = length

    @abstractmethod
    def compute_log_likelihood(self, qvec) -> float:
        """
        Compute the likelihood of a given q-vector.
        :param qvec: The query.
        :return: the marginal probability P(qvec).
        """
        pass

    @abstractmethod
    def sample_qvec(self) -> list:
        """
        Obtain a random sample.
        :return: A quality score vector from the specified distribution.
        """
        pass


class AbstractErrorModel(metaclass=ABCMeta):
    """
    Parent class for all fragment-to-read error models.
    """

    @abstractmethod
    def compute_log_likelihood(self, fragment: Fragment, read: SequenceRead) -> float:
        """
        Compute the log probability of observing the read, conditional on the fragment.
        :param fragment: The source fragment (a String)
        :param read: The read (of type SequenceRead)
        :return: the value P(read | fragment).
        """
        pass

    @abstractmethod
    def sample_noisy_read(self, fragment: str, metadata: str = "") -> SequenceRead:
        """
        Obtain a random read (q-vec and sequence pair) from a given fragment.

        :param fragment: The source fragment.
        :param metadata: The metadata to store in the read.
        :return: A list of reads sampled according to their probabilities.
        """
        pass


class AbstractTrainableErrorModel(AbstractErrorModel):
    """
    Parent class of all trainable read error models.
    """

    @abstractmethod
    def train_from_data(self, reads, fragments, iters):
        """
        Attempt to train the error model from data.
        :param reads: A list of reads from a dataset.
        :param fragments: The corresponding collection of fragments.
        :param iters: the number of iterations.
        :return: None
        """
        pass


# ======================================================================================
# ==================================== IMPLEMENTATIONS =================================
# ======================================================================================

class NoiselessErrorModel(AbstractErrorModel):
    def __init__(self, mismatch_likelihood: float = 1e-20):
        super().__init__()
        self.mismatch_log_likelihood = math.log(mismatch_likelihood)
        self.match_log_likelihood = math.log(1 - mismatch_likelihood)

    def compute_log_likelihood(self, fragment: Fragment, read: SequenceRead) -> float:
        return self.match_log_likelihood if fragment.seq == read.seq else self.mismatch_log_likelihood

    def sample_noisy_read(self, fragment: str, metadata: str = "") -> SequenceRead:
        return SequenceRead(fragment, quality=torch.ones(len(fragment))*1000, metadata=metadata)


class RampUpRampDownDistribution(AbstractQScoreDistribution):
    """
        A default (simple) deterministic implementation, assigning probability 1 to a single particular quality score
        vector.

        Assigns a slow ramp-up  and then a ramp-down of the quality values over the target length.
        The outputted quality vector has quality assignments roughly proportional to the distribution specified.
    """

    def __init__(self,
                 length: int,
                 quality_score_values: torch.Tensor,
                 distribution: torch.Tensor):
        """
        :param length: the length of q-vectors to be generated.
        :param quality_score_values: the quality score values. Should be in increasing level of quality.
        :param distribution: An array of numbers, where the ith element describes the proportion for which
            the ith quality score should appear in the quality vector.
        """
        super().__init__(length)

        if not (len(quality_score_values) == len(distribution)):
            raise ValueError("There must be exactly one ratio/frequency assigned for each quality score")

        self.distribution = distribution / distribution.sum()  # normalize to 1.
        self.quality_score_values = quality_score_values
        self.qvec = self.create_qvec()  # Hardcoded quality score vector.

    def compute_log_likelihood(self, qvec: torch.Tensor):
        """
        Likelihood is just the indicator function.
        :param qvec: the query
        :return: 1 if query is qvec, 0 else.
        """
        return int(torch.eq(qvec, self.qvec))

    def sample_qvec(self) -> torch.Tensor:
        return self.qvec

    def create_qvec(self) -> torch.Tensor:
        """
        Returns a single quality score vector ('q vector').


        The quality scores are chosen in proportion to the distribution specified
        in self.distribution.

        The current implementation assigns quality scores such that 1/2 of the specified
        frequency of the lowest score (quality_score_values[0]) is applied to the first base pairs in the fragment
        as well as the last base pairs in the fragment.

        This pattern is repeated with the next quality score (quality_score_values[1]) for the base pairs
        on both ends of the fragment that haven't been assigned a quality. The resulting quality
        pattern along the fragment is thus as follows:

            lowest QS region, medium QS region  ....     Highest QS region  ....     medium QS region,  lowest QS region

        :return: A list with length equal to the length of the input fragment list, and
        each element is a quality score from self.quality_score_values
        """

        lengths = torch.zeros(len(self.quality_score_values)*2-1)

        # Iterate over each quality score and find the length of each chunk it will
        # span in the return vector.
        for index in range(len(self.quality_score_values)):
            if index == (len(self.quality_score_values)-1):  # Midpoint. Highest quality.
                lengths[index] = self.length * self.distribution[index]
            else:
                length = self.length * self.distribution[index] / 2
                lengths[index] = length
                lengths[len(lengths)-1-index] = length

        lengths = torch.floor(lengths).to(dtype=torch.int)
        total_len = lengths.sum().item()

        if not (total_len <= self.length):
            raise ValueError("The sum of the segment lengths ({}) must be less than or equal "
                             "to the total fragment length ({})".format(total_len, self.length))

        # Allocate a fixed-length array.
        quality_vector = 10 * torch.ones(self.length)

        # Note: The quality scores at the end of the vector do not get updated because of our flooring
        # in the lengths of above.

        # Go over the quality scores from lowest to highest, and then back down from highest to lowest.
        # assigning each quality score to its dedicated chunk of positions in the vector
        # according to the lengths of each chunk calculated previously.
        cur_pos = 0
        q_idx = 0
        incr = 1

        for i, count in enumerate(lengths):
            quality_vector[cur_pos:cur_pos + count] = self.quality_score_values[q_idx]
            cur_pos = cur_pos + count
            if q_idx == self.quality_score_values.size(0) - 1:
                incr = -incr
            q_idx = q_idx + incr

        return quality_vector


# ========================================================================
# ===== Basic implementation (0~5-scale quality score) ===================
# ========================================================================

class BasicQScoreDistribution(RampUpRampDownDistribution):

    def __init__(self, length: int):
        super().__init__(
            length,
            quality_score_values=torch.tensor([0, 1, 2, 3, 4]),
            distribution=torch.tensor([0.05, 0.15, 0.30, 0.25, 0.25])
        )


class BasicErrorModel(AbstractErrorModel):
    """
    A very simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicQScoreDistribution.
    """

    # Define base change probability matrices conditioned on quality score level.

    # Example:
    # HIGH_Q_BASE_CHANGE_MATRIX[_A][_U] is the probability of observing U when the actual
    # nucleotide is _A. (e.g. each 1D array should sum to 1).

    base_indices = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    bases = ['A', 'C', 'G', 'T']

    VERYLOW_Q_BASE_CHANGE_MATRIX = torch.tensor(([0.25, 0.25, 0.25, 0.25],
                                                 [0.25, 0.25, 0.25, 0.25],
                                                 [0.25, 0.25, 0.25, 0.25],
                                                 [0.25, 0.25, 0.25, 0.25]))

    LOW_Q_BASE_CHANGE_MATRIX = torch.tensor(([0.70, 0.10, 0.10, 0.10],
                                             [0.10, 0.70, 0.10, 0.10],
                                             [0.10, 0.10, 0.70, 0.10],
                                             [0.10, 0.10, 0.10, 0.70]))

    MEDIUM_Q_BASE_CHANGE_MATRIX = torch.tensor(([0.85, 0.05, 0.05, 0.05],
                                                [0.05, 0.85, 0.05, 0.05],
                                                [0.05, 0.05, 0.85, 0.05],
                                                [0.05, 0.05, 0.05, 0.85]))

    HIGH_Q_BASE_CHANGE_MATRIX = torch.tensor(([0.91, 0.03, 0.03, 0.03],
                                              [0.03, 0.91, 0.03, 0.03],
                                              [0.03, 0.03, 0.91, 0.03],
                                              [0.03, 0.03, 0.03, 0.91]))

    VERYHIGH_Q_BASE_CHANGE_MATRIX = torch.tensor(([0.991, 0.003, 0.003, 0.003],
                                                  [0.003, 0.991, 0.003, 0.003],
                                                  [0.003, 0.003, 0.991, 0.003],
                                                  [0.003, 0.003, 0.003, 0.991]))

    Q_SCORE_BASE_CHANGE_MATRICES = [VERYLOW_Q_BASE_CHANGE_MATRIX,
                                    LOW_Q_BASE_CHANGE_MATRIX,
                                    MEDIUM_Q_BASE_CHANGE_MATRIX,
                                    HIGH_Q_BASE_CHANGE_MATRIX,
                                    VERYHIGH_Q_BASE_CHANGE_MATRIX]

    def __init__(self, read_len=150):
        self.read_len = read_len
        self.q_dist = BasicQScoreDistribution(read_len)

    def compute_log_likelihood(self, fragment: Fragment, read: SequenceRead) -> float:
        """
        Computes the log likelihood of reading 'fragment' as 'read'
        :param: read - a SequenceRead instance.
        :param: fragment
        """

        log_product = self.q_dist.compute_log_likelihood(read.quality)
        for actual_base, read_base, q_score in zip(fragment.seq, read.seq, read.quality):
            idx1 = BasicErrorModel.base_indices[actual_base]
            idx2 = BasicErrorModel.base_indices[read_base]
            base_prob = BasicErrorModel.Q_SCORE_BASE_CHANGE_MATRICES[q_score][idx1][idx2]
            log_product += torch.log(base_prob)
        return log_product

    def sample_noisy_read(self, fragment, metadata="") -> SequenceRead:
        quality_score_vector = self.q_dist.sample_qvec()

        noisy_fragment_chars = ['' for _ in range(self.read_len)]

        # Generate base pair reads from the fragment, conditioned on fragment base pairs and
        # the quality score for that base pair.
        for k in range(len(fragment)):
            # information at position k
            read_base = fragment[k]
            q_score = quality_score_vector[k].item()

            # Translate ACGT to 0,1,2,3
            read_base_index = BasicErrorModel.base_indices[read_base]

            # Look up noisy channel probabilities
            noisy_letter = torch.multinomial(
                BasicErrorModel.Q_SCORE_BASE_CHANGE_MATRICES[q_score][read_base_index],
                num_samples=1,
                replacement=True
            ).item()

            # Save the noisy output.
            noisy_fragment_chars[k] = BasicErrorModel.bases[noisy_letter]

        noisy_fragment_string = ''.join(noisy_fragment_chars)
        seq_read = SequenceRead(noisy_fragment_string, quality_score_vector, metadata=metadata)
        return seq_read


# ======================================================================
# =========== FastQ implementation =====================================
# ======================================================================

class BasicPhredScoreDistribution(RampUpRampDownDistribution):
    """
        An implementation of the quality score ramp-up ramp-down model, where the quality values are PHRED scores.
        ref: https://en.wikipedia.org/wiki/Phred_quality_score
    """

    def __init__(self, length):
        super().__init__(
            length,
            quality_score_values=torch.tensor([10, 20, 30, 40, 50]),
            distribution=torch.tensor([0.05, 0.15, 0.30, 0.25, 0.25])
        )


# class PhredScoreDistribution(AbstractQScoreDistribution):
#     # TODO: Implement something more sophisticated than a slow ramp up of and ramp down of quality
#     # TODO: over the nucleotides. Monte Carlo simulations?
#     pass


class FastQErrorModel(AbstractErrorModel):
    """
    A simple error model, based on reads of a fixed length, and q-vectors coming from an instance of
    BasicPhredScoreDistribution.
    """

    def __init__(self, read_len=150):
        self.read_len = read_len
        self.q_dist = BasicPhredScoreDistribution(read_len)

    @staticmethod
    def compute_error_prob(q: torch.Tensor) -> torch.Tensor:
        return torch.pow(10, -q.to(dtype=torch.double)/10)

    def compute_log_likelihood(self, fragment: Fragment, read: SequenceRead) -> float:
        error_prob = FastQErrorModel.compute_error_prob(read.quality)
        matches = torch.tensor([
            fragment.seq[k] == read.seq[k] for k in range(len(fragment.seq))
        ]).to(dtype=torch.double)

        return ((torch.ones(1, device=error_prob.device) - error_prob) * matches + (error_prob/3)
                * (torch.ones(1, device=error_prob.device) - matches)).log().sum().item()

    def sample_noisy_read(self, fragment: str, metadata="") -> SequenceRead:
        qvec = self.q_dist.sample_qvec()
        noisy_fragment_chars = ['' for _ in range(self.read_len)]
        error_probs = FastQErrorModel.compute_error_prob(qvec)
        error_locs = (torch.rand(size=error_probs.size()) < error_probs)
        for k in range(len(fragment)):
            if error_locs[k].item():
                noisy_fragment_chars[k] = mutate_acgt(fragment[k])
            else:
                noisy_fragment_chars[k] = fragment[k]

        return SequenceRead(''.join(noisy_fragment_chars), qvec, metadata=metadata)
