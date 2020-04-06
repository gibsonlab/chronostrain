import numpy as np
from simulation.fasta_functions import fasta_reader


"""
Matrix representing nucleotide flip errors.
M_ij = Pr(j | i), where indices (0,1,2,3) = (A,C,G,T).
"""
_default_error_model_ACGT = [
    [0.97, 0.01, 0.01, 0.01],
    [0.01, 0.97, 0.01, 0.01],
    [0.01, 0.01, 0.97, 0.01],
    [0.01, 0.01, 0.01, 0.97]
]

_ACGT_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
_ACGT_indices = ["A", "C", "G", "T"]


class NucleotideSeq:
    """
    An object representing a single annotated genome.
    """
    def __init__(self, name, seq):
        self.name = name
        self.seq = seq

    def produce_reads_poisson(self, read_len, coverage):
        """
        Samples reads from a nucleotide sequence, via a Poisson-distributed model with given intensity lambda=coverage.
        :param read_len:
        :param coverage:
        :return:
        """
        num_sites = len(self.seq) - read_len
        num_frags = np.random.poisson(coverage, size=num_sites)
        return self.produce_reads(read_len, num_frags)

    def produce_reads_multinomial(self, read_len, num_reads):
        """
        Samples reads from a nucleotide sequence, via a multinomial distribution with N=num_reads.
        Each site is weighted uniformly.
        (Note: The distribution of n iid Poisson(lambda) RVs, conditioned on their sum=N is precisely
        Multinomial(N, [1/n, ..., 1/n]).)
        :param read_len:
        :param num_reads:
        :return:
        """
        num_sites = len(self.seq) - read_len
        num_frags = np.random.multinomial(
            n=num_reads,
            pvals=[1. / num_sites for _ in range(num_sites)],
            size=1
        )
        return self.produce_reads(read_len, num_frags[0])

    def produce_reads(self, read_len, num_frags):
        """
        Produce reads from the specified number of fragments.
        Paired-end reads not implemented yet.

        TODO: implement paired end reads.

        :param read_len: length of reads.
        :param num_frags: An array of integers, one for each site along the genome (minus the edge on the right side).
        :return:
        """
        reads = [None for _ in range(num_frags.sum())]
        k = 0
        for b in range(len(num_frags)):
            for i in range(num_frags[b]):
                reads[k] = mutate_acgt(self.seq[b:b+read_len])
                k += 1
        return reads


def parse_fasta(filename):
    """
    Reads a fasta file and parses each input as a NucleotideSeq object (RNA not supported yet)
    :param filename:
    :return:
    """
    seqs = []
    for name, seq in fasta_reader(filename):
        seqs.append(NucleotideSeq(name, seq))
    return seqs


def mutate_acgt(seq, mut_matrix=_default_error_model_ACGT):
    if mut_matrix is None:
        return seq
    seq = list(seq)
    for i in range(len(seq)):
        seq[i] = np.random.choice(
            a=_ACGT_indices,
            size=1,
            p=_default_error_model_ACGT[_ACGT_dict[seq[i]]]
        )[0]
    return "".join(seq)

