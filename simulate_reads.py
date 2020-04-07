#!/bin/python3
"""
  simulate_reads.py
  Run to simulate reads from genomes specified by accession numbers.
"""

import os
import argparse
import numpy as np
from pathlib import Path
import csv

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from util.logger import logger
from database.base import AbstractStrainDatabase, SimpleCSVStrainDatabase
from model.bacteria import Population
from model.reads import SequenceRead, FastQErrorModel
from model.generative import GenerativeModel, softmax

from typing import List


_data_dir = "data"


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

    parser.add_argument('-t', '--time_points', required=True, type=int, nargs="+",
                        help='<Required> A list of intergers, where each integer represents a time point'
                             'for which to take a sample from. Time points are saved as part of file name.')

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
                        help='<Required> A csv containing the relatively abundances for each strain by time point.')
    parser.add_argument('-p', '--out_prefix', required=False, default='sampled_read',
                        help='<Optional> File prefix for the read files.')
    parser.add_argument('-e', '--extension', required=False, default='txt',
                        help='<Optional> File extension.')

    return parser.parse_args()


def save_to_fastq(sampled_reads: List[List[SequenceRead]], time_points: List[int], out_dir: str, out_prefix: str):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if len(sampled_reads) != len(time_points):
        raise ValueError("Number of time indexed lists of reads should equal number of time points to read at")

    for i, t in enumerate(time_points):
        filename = '{}_t{}.fastq'.format(out_prefix, t)
        out_path = os.path.join(out_dir, filename)
        save_timeslice_to_fastq(sampled_reads[i], out_path)


def save_timeslice_to_fastq(timeslice_reads: List[SequenceRead], out_path: str):

    # Save reads taken at a particular timepoint to fastq using SeqIO library
    records = []
    for i, read in enumerate(timeslice_reads):

        # https://biopython.org/docs/1.74/api/Bio.SeqRecord.html
        record = SeqRecord(Seq(read.seq), id="Read number " + str(i), description=", a simulated read")
        record.letter_annotations["phred_quality"] = list(read.quality)

        records.append(record)

    SeqIO.write(records, out_path, "fastq")


def sample_reads(population: Population,
                 abundances: List[np.array],
                 read_depths: List[int],
                 read_length: int,
                 time_points: List[int],
                 seed: int = 31415) -> List[List[SequenceRead]]:
    np.random.seed(seed)

    ##############################
    # Construct generative model

    mu = np.array([0] * len(population.strains))  # One dimension for each strain
    tau_1 = 1
    tau = 1

    my_error_model = FastQErrorModel(read_len=read_length)

    my_model = GenerativeModel(times=time_points,
                               mu=mu,
                               tau_1=tau_1,
                               tau=tau,
                               bacteria_pop=population,
                               read_length=read_length,
                               read_error_model=my_error_model)

    ##############################
    # Generate trajectory if not already given and then sample.
    if abundances:
        for abundance_profile in abundances:
            if len(abundance_profile) != len(population.strains):
                raise ValueError("Length of abundance profiles ({}) must match number of strains. ({})".
                                 format(len(abundance_profile), len(population.strains)))
        if len(abundances) != len(time_points):
            raise ValueError("Number of abundance profiles ({}) must match number of time points ({}).".
                             format(len(abundances), len(time_points)))

        normalized_abundances = []
        for Z in abundances:
            normalized_abundances.append(softmax(Z))
        time_indexed_reads = my_model.sample_timed_reads(normalized_abundances, read_depths)
    else:
        abundances, time_indexed_reads = my_model.sample_abundances_and_reads(read_depths)

    return time_indexed_reads


def get_abundances(file: str) -> List[np.array]:
    """
    Read time-indexed abundances from file.
    :param file:
    :return: a time indexed list of abundance profiles. Each element is a list itself containing the relative abundances
    of strains at a particular time point.
    """

    file_path = os.path.join(_data_dir, file)
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        strain_abundances = [np.array(row, dtype='float') for i, row in enumerate(reader) if i != 0]

    return strain_abundances


def load_genome_database(accession_csv_file: str) -> AbstractStrainDatabase:

    # To reflect real-world data collection/fastq file generation, we let the entire genome be a single marker
    # so that every l-length fragment in the genome has a chance of being sampled, regardless of whether its
    # in an actual marker region for a strain.

    # The SimpleCSVStrainDatabase object makes this one marker per species with that one marker
    # containing the entire genome, just as we want.

    database_obj = SimpleCSVStrainDatabase(accession_csv_file)
    return database_obj


def parse_population(strain_db: AbstractStrainDatabase, accession_csv_file: str) -> Population:
    """
    Creates a Population object after finding markers for each strain listed in accession_csv_file.
    """
    file_path = os.path.join(_data_dir, accession_csv_file)

    strains = []
    # for each strain_id, create a Strain instance
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            strain_id = row['Accession']
            strain = strain_db.get_strain(strain_id)
            strains.append(strain)
    return Population(strains)


def main():
    try:
        logger.info("Pipeline for read simulation started.")
        args = parse_args()
        genome_database = load_genome_database(args.accession_file)
        population = parse_population(genome_database, args.accession_file)

        abundances = None
        if args.abundance_file:
            logger.debug("Parsing abundance file...")
            abundances = get_abundances(file=args.abundance_file)

        logger.debug("Sampling reads...")
        time_points = args.time_points
        read_depths = args.num_reads * np.ones(len(time_points), dtype=int)
        sampled_reads = sample_reads(
            population=population,
            read_depths=read_depths,
            abundances=abundances,
            read_length=args.read_length,
            time_points=time_points,
            seed=args.seed
        )

        logger.debug("Saving samples to FastQ file {}.".format(args.out_dir + "/" + args.out_prefix))
        save_to_fastq(sampled_reads, args.time_points, args.out_dir, args.out_prefix)
        logger.info("Reads finished sampling.")
    except Exception as e:
        logger.error("Uncaught exception -- {}".format(e))


if __name__ == "__main__":
    main()
