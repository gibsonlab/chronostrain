"""
  inference.py
  Run to perform inference on specified reads.
"""
import argparse
from util.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # =============
    # TODO: add arguments to parser.
    # =============
    return parser.parse_args()


def load_marker_database():
    # TODO -- could just return a default implementation for now.
    raise NotImplementedError("TODO implement!")


def load_from_fastq(filenames):
    num_times = len(filenames)
    logger.debug("Number of time points: {}".format(num_times))
    # TODO -- read fastQ files.
    raise NotImplementedError("TODO implement!")


def perform_inference(args):
    # TODO -- call EM algorithm here.
    raise NotImplementedError("TODO implement!")


def main():
    logger.info("Pipeline for inference started.")
    args = parse_args()
    logger.debug("Downloading marker database.")
    load_marker_database()
    logger.debug("Reading time-series read files.")
    reads = load_from_fastq(args.read_files)
    logger.debug("Performing inference.")
    abundances = perform_inference(reads)
    logger.info(str(abundances))  # Should output this to its own separate file.
    logger.info("Inference finished.")


if __name__ == "__main__":
    main()

