"""
    Contains abstract classes. See the other python files for implementations.
"""
from typing import Union

import numpy as np
from abc import abstractmethod, ABCMeta
from chronostrain.model import Fragment
from chronostrain.util.sequences import nucleotides_to_z4, z4_to_nucleotides, SeqType, reverse_complement_seq


class SequenceRead:
    """
    A class representing a sequence-quality vector pair.
    """
    def __init__(self, read_id: str, seq: Union[str, SeqType], quality: np.ndarray, metadata: str):
        self.id: str = read_id
        if len(seq) != len(quality):
            raise ValueError(
                "Length of nucleotide sequence ({}) must agree with length of quality score sequence ({})".format(
                    len(seq), len(quality)
                )
            )

        """
        The sequence content of the read is stored as a numpy-optimized array of ubyte.
        """
        if isinstance(seq, str):
            self.seq: np.ndarray = nucleotides_to_z4(seq)
        elif isinstance(seq, np.ndarray):
            self.seq = seq
        else:
            raise TypeError("Unexpected type for argument `seq` (got `{}`)".format(type(seq)))
        self.quality: np.array = quality
        self.metadata: str = metadata

    def nucleotide_content(self, reverse_complement: bool = False) -> str:
        if reverse_complement:
            return z4_to_nucleotides(reverse_complement_seq(self.seq))
        else:
            return z4_to_nucleotides(self.seq)

    def __str__(self):
        return "[SEQ:{},QUAL:{}]".format(
            z4_to_nucleotides(self.seq),
            self.quality
        )

    def __repr__(self):
        return "[SEQ:{},QUAL:{}]".format(
            self.seq,
            self.quality
        )

    def __len__(self):
        return self.seq.shape[0]

    def __eq__(self, other):
        if not isinstance(other, SequenceRead):
            return False
        return other.id == self.id

    def __hash__(self):
        return hash(self.id)


class AbstractErrorModel(metaclass=ABCMeta):
    """
    Parent class for all fragment-to-read error models. This class (and its implementations) determines the
    likelihood values of the model (and therefore all inference algorithms).
    """

    @abstractmethod
    def compute_log_likelihood(self,
                               fragment: Fragment,
                               read: SequenceRead,
                               read_reverse_complemented: bool,
                               insertions: np.ndarray,
                               deletions: np.ndarray,
                               read_start_clip: int = 0,
                               read_end_clip: int = 0) -> float:
        """
        Compute the log probability of observing the read, conditional on the fragment.
        :param fragment: The source fragment (a String)
        :param read: The read (of type SequenceRead)
        :param read_reverse_complemented: Indicates whether the read ought to be reverse complemented.
        :param insertions: A boolean array that indicates inserted nucleotides of the read.
            The length of the array must match length of read. (extra characters not in fragment).
        :param deletions: A boolean array that indicates deleted nucleotides from the fragment
            The length of the array much match length of fragment. (characters missing in read from fragment).
        :param read_start_clip: The number of bases to clip off the beginning of the read.
        :param read_end_clip: The number of bases to clip off the end of the read.
        :return: the value P(read | fragment).

        Note: If `reverse complement` is specified, this will apply first BEFORE applying insertions/deletions/clipping.
        """
        pass

    @abstractmethod
    def sample_noisy_read(self, read_id: str, fragment: Fragment, metadata: str = "") -> SequenceRead:
        """
        Obtain a random read (q-vec and sequence pair) from a given fragment.

        :param read_id: The ID of the read.
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


class AbstractQScoreDistribution(metaclass=ABCMeta):
    """
    Parent class for all Q-score distributions. Meant as a utility function to help compute log likelihoods in
    error models (AbstractErrorModel).
    """

    @abstractmethod
    def compute_log_likelihood(self, qvec: np.ndarray) -> float:
        """
        Compute the likelihood of a given q-vector.
        :param qvec: The query.
        :return: the marginal probability P(qvec).
        """
        pass

    @abstractmethod
    def sample_qvec(self) -> np.ndarray:
        """
        Obtain a random sample.
        :return: A quality score vector from the specified distribution.
        """
        pass
