#!/bin/python3
"""
  simulate_reads.py
  Run to simulate reads from genomes specified by accession numbers.
"""

import os
import argparse
import re
import csv
import torch

from util.io.logger import logger

from typing import List, Tuple
from database.base import SimpleCSVStrainDatabase

from model import generative, reads
from model.bacteria import Population
from model.reads import SequenceRead
from util.io.model_io import save_reads_to_fastq, save_abundances

_data_dir = "data"

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate reads from genomes.")

    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> Directory to save the reads.')

    parser.add_argument('-a', '--accession_file', required=True, type=str,
                        help='<Required> File listing the species to sample from. Should be in csv format as follows:'
                             '"Name","Accession" \
                              "Clostridium sporogenes ATCC 15579","NZ_DS981518.1" \
                              "Enterococcus faecalis V583","NC_004668.1" \
                              "Bacteroides fragilis NCTC9343","CR626927.1"')

    parser.add_argument('-n', '--num_reads', required=True, type=int, nargs='+',
                        help='<Required> Numbers of the reads to sample at each time point. '
                             'Must either be a single integer or a list of integers with length equal to the '
                             'number of time points.')

    parser.add_argument('-l', '--read_length', required=True, type=int,
                        help='<Required> Length of the reads to sample.')

    # Optional params
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-b', '--abundance_file', required=False, type=str,
                        help='<Optional> A csv containing the relatively abundances for each strain by time point.')
    parser.add_argument('-t', '--time_points', required=False, type=int, nargs="+",
                        help='<Optional; Required if -b not specified> A list of integers. Each value represents '
                             'a time point in the dataset. Time points are saved as part of file name.')
    parser.add_argument('-p', '--out_prefix', required=False, default='sim',
                        help='<Optional> File prefix for the read files.')
    parser.add_argument('-trim', '--marker_trim_len', required=False, type=int,
                        help='<Optional> An integer to trim markers down to. For testing/debugging.')

    return parser.parse_args()


def sample_reads(
        population: Population,
        read_depths: List[int],
        read_length: int,
        time_points: List[int],
        abundances: torch.Tensor = None,
        seed: int = 31415) -> Tuple[torch.Tensor, List[List[SequenceRead]]]:
    """
    Sample sequence reads from the generative model, using either a pre-specified abundance profile or using
    random samples.

    :param population: The population containing the Strain instances.
    :param read_depths: The read counts for each time point.
    :param read_length: The read length.
    :param time_points: A list of time values (in increasing order).
    :param abundances: (Optional) An abundance profile as a T x S tensor.
     Could be positive-valued weights (e.g. absolute abundances).
     If none specified, the generative model samples its own from a Gaussian process.
    :param seed: (Optional, default:31415) The random seed to use for sampling (to encourage reproducibility).
    :return: (1) The relative abundance profile and (2) the sampled reads (time-indexed).
    """
    torch.manual_seed(seed)

    # Default/unbiased parameters for prior.
    mu = torch.zeros(len(population.strains), device=default_device)  # One dimension for each strain
    tau_1 = 1
    tau = 1

    # Construct a GenerativeModel instance.
    my_error_model = reads.FastQErrorModel(read_len=read_length)
    # my_error_model = reads.NoiselessErrorModel()
    my_model = generative.GenerativeModel(times=time_points,
                                          mu=mu,
                                          tau_1=tau_1,
                                          tau=tau,
                                          bacteria_pop=population,
                                          read_length=read_length,
                                          read_error_model=my_error_model,
                                          torch_device=default_device)

    if len(read_depths) != len(time_points):
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


def get_abundances(file: str) -> Tuple[List[int], torch.Tensor, List[str]]:
    """
    Read time-indexed abundances from file.
    :param file: The filename with abundances.
    :return: (1) A list of time points,
    (2) a time indexed list of abundance profiles,
    (3) the list of relevant accessions.
    """

    time_points = []
    strain_abundances = []
    accessions = []

    file_path = os.path.join(_data_dir, file)
    with open(file_path, newline='') as f:
        reader = csv.reader(f, quotechar='"')
        for i, row in enumerate(reader):
            if i == 0:
                accessions = [accession.replace('"', '').strip() for accession in row[1:]]
                continue
            if not row:
                continue
            time_point = row[0]
            abundances = torch.tensor(
                [float(val) for val in row[1:]],
                dtype=torch.double,
                device=default_device
            )
            time_points.append(time_point)
            strain_abundances.append(abundances)
    return time_points, torch.stack(strain_abundances, dim=0), accessions


def get_genomes(accession_nums, strain_info):
    """
    For each accession num, retrieve genome info from strain_info.
    """

    genomes_map = {}

    for accession_num in accession_nums:
        if accession_num in strain_info.keys():
            filename = os.path.join(_data_dir, accession_num + ".fasta")
            with open(filename) as file:
                for i, line in enumerate(file):
                    genome = re.sub('[^AGCT]+', '', line.split(sep=" ")[-1])
            genomes_map[accession_num] = genome
    return genomes_map


def main():
    logger.info("Pipeline for read simulation started.")
    args = parse_args()

    # ==============================================
    # Note: The usage of "SimpleCSVStrainDatabase" initializes the strain information so that each strain's (unique)
    # marker is its own genome.
    # ==============================================
    database = SimpleCSVStrainDatabase(args.accession_file, trim_debug=args.marker_trim_len)

    # ========= Load abundances and accessions.
    abundances = None
    accessions = None
    if args.abundance_file:
        logger.debug("Parsing abundance file...")
        time_points, abundances, accessions = get_abundances(file=args.abundance_file)
    else:
        time_points = args.time_points

    if time_points is None:
        raise Exception("(Time points) argument is required if abundances file not specified.")

    # ========== Create Population instance.
    if accessions:
        print(database.get_strains(accessions))
        population = Population(database.get_strains(accessions), torch_device=default_device)
    else:
        population = Population(database.all_strains(), torch_device=default_device)

    # ========== Sample reads.
    logger.debug("Sampling reads...")
    abundances, sampled_reads = sample_reads(
        population=population,
        read_depths=args.num_reads,
        abundances=abundances,
        read_length=args.read_length,
        time_points=time_points,
        seed=args.seed
    )

    # ========== Save sampled reads to file.
    logger.debug("Saving samples to file...")
    save_reads_to_fastq(sampled_reads, time_points, args.out_dir, args.out_prefix)
    logger.debug("Saving abundances to file...")
    save_abundances(
        population,
        time_points,
        abundances,
        '{}_abundances.csv'.format(args.out_prefix),
        out_dir=args.out_dir
    )
    logger.info("Reads finished sampling.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
