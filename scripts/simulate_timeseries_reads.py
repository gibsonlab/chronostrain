import numpy as np
import random

def generate_random_abundances(times, num_strains, scale=1.0):
    abundances = []
    previous_t = None
    for t in times:
        if len(abundances) == 0:
            abundances.append(random_vector(num_strains))
        else:
            next_value = np.random.multivariate_normal(
                mean=abundances[-1],
                cov=np.eye(num_strains) * scale * (t - previous_t),
            )
            abundances.append(next_value)
        previous_t = t
    return np.transpose(np.stack(abundances))


def random_vector(size):
    vec = np.random.uniform(low=1.0, high=100.0, size=size)
    return vec


def simulate_reads(genomes, timeseries_abundances, timeseries_num_reads, read_len):
    """
    Simulate metagenomic reads according to specified abundances.
    :param genomes: A List of NucleotideSeq Objects, one per strain.
    :param timeseries_abundances: A time-indexed array of strain-indexed (numpy) array of real values.
    :param timeseries_num_reads: A time-indexed array of real values, which determines the total number of reads per time slice.
    :return:
    """
    if len(genomes) != len(timeseries_abundances[0]):
        raise RuntimeError("Number of genomes should match the number of strains. Got: {}, {}".format(
            len(genomes),
            len(timeseries_abundances[0])
        ))

    if len(timeseries_abundances) != len(timeseries_num_reads):
        raise RuntimeError("Timeseries arrays should be equal lengths. Got: {}, {}".format(
            len(timeseries_abundances),
            len(timeseries_num_reads)
        ))

    timeseries_reads = []
    for t in range(len(timeseries_abundances)):
        # Pick the number of reads for each strain at time t
        num_reads = timeseries_num_reads[t]
        abundances = timeseries_abundances[t]
        strain_num_reads = np.random.multinomial(n=num_reads, pvals=abundances / abundances.sum(), size=1)[0]
        reads_t = []
        for k in range(len(genomes)):
            # Sample the reads from each strain.
            genome = genomes[k]
            count = strain_num_reads[k]
            reads_t += genome.produce_reads_multinomial(read_len=read_len, num_reads=count)
        random.shuffle(reads_t)
        timeseries_reads.append(reads_t)
    return timeseries_reads
