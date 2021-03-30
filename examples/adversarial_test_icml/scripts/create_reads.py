"""
Implement the read sampling scheme, adversarially chosen, as described in the ICML poster.

Figure caption:

Figure 3.A plot showing the outcome of synthetic experiments with a constant number (5) of corrupted reads.
The goal is to estimate the single time-point abundance of two strains A and B differing by exactly one
nucleotide.  However, five of the reads from A were corrupted to have B’s SNV with low quality at that base.
Incorporation of quality information drastically affects the accuracy of the inference, especially when one
has fewer reads to work with.
"""
import os
import argparse
import math
import numpy as np
from chronostrain.model.reads import SequenceRead
from chronostrain.model.io import save_timeslice_to_fastq


def parse_args():
    parser = argparse.ArgumentParser(description="Sample reads, five of which are corrupted.")
    parser.add_argument('-r', '--reads_dir', required=True, type=str,
                        help='<Required> Directory to store read files.')
    return parser.parse_args()


def make_reads(num_reads_per_block: int, num_corrupted_reads: int):
    corrupted_quality = np.array(
        [1, 5, 8, 10, 15] + [17]*5 + [20]*5 + [25]*5 + [30]*5 + [40]*10 + [35]*5 + [30]*5 + [25]*5,
        dtype=np.float
    )
    regular_quality = np.array(
        [10, 11, 12, 13, 15] + [17]*5 + [20]*5 + [25]*5 + [30]*5 + [40]*10 + [35]*5 + [30]*5 + [25]*5,
        dtype=np.float
    )

    corrupted_read = SequenceRead(seq='ACTTTATGGTTATGAATTTGCTATTGCCGGTGAATTATTGAGGGATTATG',
                                  quality=corrupted_quality,
                                  metadata='A:pos1(corrupted)')
    uncorrupted_read_a = SequenceRead(seq='TCTTTATGGTTATGAATTTGCTATTGCCGGTGAATTATTGAGGGATTATG',
                                      quality=regular_quality,
                                      metadata='A:pos1')
    uncorrupted_read_b = SequenceRead(seq='ACTTTATGGTTATGAATTTGCTATTGCCGGTGAATTATTGAGGGATTATG',
                                      quality=regular_quality,
                                      metadata='B:pos1')

    num_a = math.ceil(num_reads_per_block * 0.8)
    num_uncorrupted_a = max(num_a - num_corrupted_reads, 0)
    num_b = num_reads_per_block - (num_corrupted_reads + num_uncorrupted_a)

    reads_0 = (
            [corrupted_read for _ in range(num_corrupted_reads)]
            + [uncorrupted_read_a for _ in range(num_uncorrupted_a)]
            + [uncorrupted_read_b for _ in range(num_b)]
    )

    print("num_reads_per_block = ", num_reads_per_block)
    print("# corrupted A = ", num_corrupted_reads)
    print("# uncorrupted A = ", num_uncorrupted_a)
    print("# B = ", num_b)

    reads_1 = [
        SequenceRead(seq='CTTAAAGTGTTCTATCCTGCTAATGATGATTTCCTGAAACGTCATCACGA',
                     quality=regular_quality, metadata='AB:pos95')
        for _ in range(num_reads_per_block)
    ]

    reads_2 = [
        SequenceRead(seq='TGCGTTACATTATTGGGCTAACTGGTGTCTTTGTAATATAGCGGCTAAAA',
                     quality=regular_quality, metadata='AB:pos151')
        for _ in range(num_reads_per_block)
    ]

    reads_3 = [
        SequenceRead(seq='GCCGGTGAATTATTGAGGGATTATGGAGGATGGGATCGTGCGGACTTCGC',
                     quality=regular_quality, metadata='AB:pos26')
        for _ in range(num_reads_per_block)
    ]

    return reads_0 + reads_1 + reads_2 + reads_3


def save_input_csv(out_dir, read_file):
    with open(os.path.join(out_dir, 'input_files.csv'), "w") as f:
        print("\"1\",\"{}\"".format(read_file), file=f)


def main():
    args = parse_args()
    num_corrupted_reads = 5

    for num_reads_per_block in range(5, 55, 5):
        block_dir = os.path.join(args.reads_dir, "depth_{}".format(4*num_reads_per_block))
        if not os.path.exists(block_dir):
            os.makedirs(block_dir)

        read_filepath = os.path.abspath(os.path.join(block_dir, "reads.fastq"))
        reads = make_reads(num_reads_per_block, num_corrupted_reads)
        save_timeslice_to_fastq(reads, read_filepath)

        save_input_csv(out_dir=block_dir, read_file=read_filepath)


if __name__ == "__main__":
    main()
