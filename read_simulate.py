"""
  read_simulate.py
  Run to simulate reads from genomes specified by accession numbers.
"""
import argparse
from scripts.fetch_genomes import fetch_sequences
from util.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate reads from genomes.")

    # =============
    # TODO: add arguments to parser.
    # =============
    return parser.parse_args()


def save_to_fastq(sampled_reads):
    # TODO
    raise NotImplementedError("TODO implement!")


def sample_reads(args):
    # TODO
    raise NotImplementedError("TODO implement!")


def main():
    logger.info("Pipeline for read simulation started.")
    args = parse_args()
    logger.debug("Downloading genomes from NCBI...")
    fetch_sequences()
    logger.debug("Sampling reads...")
    sampled_reads = sample_reads(args)
    logger.debug("Saving samples to FastQ file {}.".format(args.out_file))
    save_to_fastq(args, sampled_reads)
    logger.info("Reads finished sampling.")



if __name__ == "__main__":
    main()

