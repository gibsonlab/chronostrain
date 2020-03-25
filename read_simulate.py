"""
  read_simulate.py
  Run to simulate reads from genomes specified by accession numbers.
"""
import os
import argparse
from scripts.fetch_genomes import fetch_sequences
from util.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate reads from genomes.")
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-o', '--out_dir', required=True,
                        help='<Required> Directory to save the reads.')
    parser.add_argument('-p', '--out_prefix', required=False, default='sampled_read',
                        help='<Optional> File prefix for the read files.')
    parser.add_argument('-e', '--extension', required=False, default='txt',
                        help='<Optional> File extension.')

    # =============
    # TODO: add arguments to parser.
    # =============
    return parser.parse_args()


def save_to_fastq(sampled_reads, out_dir, out_prefix, extension):
    for t in range(len(sampled_reads)):
        filename = '{}_{}.{}'.format(out_prefix, t, extension)
        out_path = os.path.join(out_dir, filename)
        save_timeslice_to_fastq(sampled_reads[t], out_path)


def save_timeslice_to_fastq(reads, out_path):
    # TODO implement.
    pass


def sample_reads(param_1, param_2, seed):
    # TODO
    raise NotImplementedError("TODO implement!")


def main():
    logger.info("Pipeline for read simulation started.")
    args = parse_args()
    logger.debug("Downloading genomes from NCBI...")
    fetch_sequences()
    logger.debug("Sampling reads...")
    sampled_reads = sample_reads(
        param_1=args.p1,  # Change this as necessary.
        param_2=args.p2,  # Change this as necessary.
        seed=args.seed
    )
    logger.debug("Saving samples to FastQ file {}.".format(args.out_file))
    save_to_fastq(sampled_reads, args.out_dir, args.out_prefix)
    logger.info("Reads finished sampling.")



if __name__ == "__main__":
    main()

