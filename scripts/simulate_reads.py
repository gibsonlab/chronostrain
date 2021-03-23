#!/bin/python3
"""
  simulate_reads.py
  Run to simulate reads from genomes specified by raccession numbers.
"""

import argparse
import os

import torch
from typing import List, Tuple

from chronostrain import logger, cfg
from chronostrain.database import StrainNotFoundError
from chronostrain.model import generative, reads
from chronostrain.model.bacteria import Population
from chronostrain.model.reads import SequenceRead
from chronostrain.model.io import save_reads_to_fastq, save_abundances, load_abundances


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate reads from genomes.")

    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> Directory to save the reads.')

    parser.add_argument('-n', '--num_reads', required=True, type=int, nargs='+',
                        help='<Required> Numbers of the reads to sample at each time point. '
                             'Must either be a single integer or a list of integers with length equal to the '
                             'number of time points.')

    parser.add_argument('-l', '--read_length', required=True, type=int,
                        help='<Required> Length of the reads to sample.')

    parser.add_argument('-b', '--abundance_path', required=False, type=str,
                        help='<Required if -t not specified> '
                             'A path to a CSV file containing the time-indexed relative abundances for each strain.')

    parser.add_argument('-t', '--time_points', required=False, type=float, nargs="+",
                        help='<Required if -b not specified> '
                             'A list of integers. Each value represents a time point in the dataset.')

    # Optional params
    parser.add_argument('-s', '--seed', required=False, type=int, default=None,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-p', '--out_prefix', required=False, default='sim',
                        help='<Optional> File prefix for the read files. '
                             'Files are saved in the format [PREFIX]_reads_t[TIME].fastq. (Default: "sim")')

    return parser.parse_args()


def sample_reads(
        population: Population,
        read_depths: List[int],
        read_length: int,
        time_points: List[float],
        disable_quality: bool,
        abundances: torch.Tensor = None,
        seed: int = None) -> Tuple[torch.Tensor, List[List[SequenceRead]]]:
    """
    Sample sequence reads from the generative model, using either a pre-specified abundance profile or using
    random samples.

    :param population: The population containing the Strain instances.
    :param read_depths: The read counts for each time point.
    :param read_length: The read length.
    :param time_points: A list of time values (in increasing order).
    :param disable_quality: A flag to indicate whether to use NoiselessErrorModel.
    :param abundances: (Optional) An abundance profile as a T x S tensor.
     Could be positive-valued weights (e.g. absolute abundances).
     If none specified, the generative model samples its own from a Gaussian process.
    :param seed: (Optional, default:31415) The random seed to use for sampling (to encourage reproducibility).
    :return: (1) The relative abundance profile and (2) the sampled reads (time-indexed).
    """
    if seed:
        torch.manual_seed(seed)

    # Default/unbiased parameters for prior.
    mu = torch.zeros(len(population.strains) - 1, device=cfg.torch_cfg.device)  # One dimension for each strain
    tau_1 = 1
    tau = 1

    # Construct a GenerativeModel instance.
    if disable_quality:
        logger.info("Flag --disable_quality turned on; Quality scores are diabled.")
        my_error_model = reads.NoiselessErrorModel()
    else:
        my_error_model = reads.BasicFastQErrorModel(read_len=read_length)
    my_model = generative.GenerativeModel(times=time_points,
                                          mu=mu,
                                          tau_1=tau_1,
                                          tau=tau,
                                          bacteria_pop=population,
                                          read_length=read_length,
                                          read_error_model=my_error_model)

    if len(read_depths) != len(time_points):
        logger.warning("Not enough read depths (len={}) specified for time points (len={}). "
                       "Defaulting to {} per time point.".format(len(read_depths), len(time_points), read_depths[0]))
        read_depths = [read_depths[0]]*len(time_points)

    if abundances is not None:
        # If abundance profile is provided, normalize it and interpret that as the relative abundance.
        for abundance_profile in abundances:
            if len(abundance_profile) != len(population.strains):
                raise ValueError("Length of abundance profiles ({}) must match number of strains. ({})".
                                 format(len(abundance_profile), len(population.strains)))
        if len(abundances) != len(time_points):
            raise ValueError("Number of abundance profiles ({}) must match number of time points ({}).".
                             format(len(abundances), len(time_points)))

        logger.info("Generating sample reads from specified ({} x {}) abundance profile.".format(
            abundances.size(0), abundances.size(1)
        ))
        abundances = abundances / abundances.sum(dim=1, keepdim=True)
        time_indexed_reads = my_model.sample_timed_reads(abundances, read_depths)
    else:
        # Otherwise, sample our own abundances.
        logger.info("Sampling ({} x {}) abundance profile and reads.".format(
            my_model.num_times(), my_model.num_strains()
        ))
        abundances, time_indexed_reads = my_model.sample_abundances_and_reads(read_depths)

    return abundances, time_indexed_reads


def save_input_csv(time_points, out_dir, out_filename, read_files):
    with open(os.path.join(out_dir, out_filename), "w") as f:
        for t, read_file in zip(time_points, read_files):
            print("\"{}\",\"{}\"".format(t, read_file), file=f)


def main():
    logger.info("Read simulation started.")
    args = parse_args()
    database = cfg.database_cfg.get_database()

    # ========= Load abundances and accessions.
    abundances = None
    accessions = None
    if args.abundance_path:
        logger.debug("Parsing abundance file...")
        time_points, abundances, accessions = load_abundances(
            file_path=args.abundance_path
        )
    else:
        time_points = args.time_points

    if time_points is None:
        raise Exception("(Time points -t) argument is required if abundances file (-b) not specified.")

    # ========== Create Population instance.
    if accessions:
        try:
            population = Population(database.get_strains(accessions))
        except StrainNotFoundError as e:
            print("Strain `{}` from abundances file `{}` not found in database.".format(
                e.strain_id,
                args.abundance_path
            ))
            raise e
    else:
        population = Population(database.all_strains())

    # ========== Sample reads.
    logger.debug("Sampling reads...")
    abundances, sampled_reads = sample_reads(
        population=population,
        read_depths=args.num_reads,
        abundances=abundances,
        read_length=args.read_length,
        disable_quality=(not cfg.model_cfg.use_quality_scores),
        time_points=time_points,
        seed=args.seed
    )

    # ========== Save sampled reads to file.
    logger.debug("Saving samples to file...")
    read_files = save_reads_to_fastq(sampled_reads, time_points, args.out_dir, args.out_prefix)
    logger.debug("Saving abundances to file...")
    save_abundances(
        population=population,
        time_points=time_points,
        abundances=abundances,
        out_dir=args.out_dir,
        out_filename='{}_abundances.csv'.format(args.out_prefix)
    )

    save_input_csv(
        time_points=time_points,
        out_dir=args.out_dir,
        out_filename='input_files.csv',
        read_files=read_files
    )


if __name__ == "__main__":
    main()
