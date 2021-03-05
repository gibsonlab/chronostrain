import torch
from chronostrain.model import Fragment
from chronostrain.model.reads.base import AbstractErrorModel, SequenceRead
from chronostrain.model.reads.basic import RampUpRampDownDistribution
from chronostrain.model.reads._util import mutate_acgt


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


class BasicFastQErrorModel(AbstractErrorModel):
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
        error_prob = BasicFastQErrorModel.compute_error_prob(read.quality)
        matches = torch.tensor([
            fragment.seq[k] == read.seq[k] for k in range(len(fragment.seq))
        ]).to(dtype=torch.double)

        return ((torch.ones(1, device=error_prob.device) - error_prob) * matches +
                (error_prob/3) * (torch.ones(1, device=error_prob.device) - matches)).log().sum().item()

    def sample_noisy_read(self, fragment: str, metadata="") -> SequenceRead:
        qvec = self.q_dist.sample_qvec()
        noisy_fragment_chars = ['' for _ in range(self.read_len)]
        error_probs = BasicFastQErrorModel.compute_error_prob(qvec)
        error_locs: torch.Tensor = (torch.rand(size=error_probs.size()) < error_probs)
        for k in range(len(fragment)):
            if error_locs[k].item():
                noisy_fragment_chars[k] = mutate_acgt(fragment[k])
            else:
                noisy_fragment_chars[k] = fragment[k]

        return SequenceRead(''.join(noisy_fragment_chars), qvec, metadata=metadata)
