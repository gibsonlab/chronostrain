#!/bin/python3
"""
  run_inference.py
  Run to perform inference on specified reads.
"""

import random, argparse
import numpy as np
import pandas as pd
from util.logger import logger
from database.base import *

from Bio import SeqIO

from model.bacteria import Population, Strain
from model import generative
from model.reads import FastQErrorModel
from algs import em, vi


_data_dir = "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")
    parser.add_argument('-d', '--read_files_dir', required=True,
                        help='<Required> Directory containing read files.')
    parser.add_argument('-r', '--read_files', nargs='+', required=True,
                        help='<Required> One read file per time point (minimum 1)')
    parser.add_argument('-a', '--accession_file', required=True, type=str,
                        help='<Required> File listing the species to sample from. Should be in csv format as follows:'
                             '"Name","Accession" \
                              "Clostridium sporogenes ATCC 15579","NZ_DS981518.1" \
                              "Enterococcus faecalis V583","NC_004668.1" \
                              "Bacteroides fragilis NCTC9343","CR626927.1"')
    parser.add_argument('-t', '--time_points', nargs='+', type=int,
                        help='List of time points.')
    parser.add_argument('-l', '--read_length', required=True, type=int,
                        help='<Required> Length of the reads to sample.')
    parser.add_argument('-m', '--method', choices=['em', 'vi', 'bbvi'], required=True,
                        help='<Required> Inference method.')

    # Optional params
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')

    return parser.parse_args()


def load_marker_database(accession_csv_file : str) -> AbstractDatabase:
    database_obj = SimpleCSVDatabase(accession_csv_file)
    database_obj.load()
    return database_obj


def parse_population(strain_db: AbstractDatabase, accession_csv_file : str) -> Population:
    """
    Creates a Population object after finding markers for each strain listed in accession_csv_file.
    """
    file_path = os.path.join(_data_dir, accession_csv_file)
    accession_csv_df = pd.read_csv(file_path)

    strains = []
    # for each strain_id, create a Strain instance
    for strain_id in accession_csv_df['Accession'].tolist():
        markers = strain_db.get_markers(strain_id)

        # markers = [marker[1:30] for marker in markers] # TODO:Only for debugging. Make sure cmd line win size param<30
        strain_instance = Strain(name=strain_id, markers=markers)
        strains.append(strain_instance)

    return Population(strains)


def load_from_fastq(file_dir_name: str, filenames: List[str]) -> List[List[generative.SequenceRead]]:
    """
    param file_dir_name - Name of directory containing the fastq files.
    param filesnames - a list of strings of filenames. Must be in chronological order.
            filenames[0] == name of fastq file for first time point
            filenames[1] == name of fastq file for second time point
            etc...
    return - A list of lists, where the ith inner list contains all of the reads taken at the ith time point.
    """

    num_times = len(filenames)
    logger.debug("Number of time points: {}".format(num_times))

    # Parse the reads (include quality)
    reads = []  # A time-indexed list of read sets. Each entry is itself a list of reads for time t.
    for file in filenames:

        reads_at_t = []  # A list of reads at a particular time (i.e. the reads in 'file')
        file_path = os.path.join(file_dir_name, file)
        for record in SeqIO.parse(file_path, "fastq"):
            read = generative.SequenceRead(seq=str(record.seq),
                                           quality=np.asanyarray(record.letter_annotations["phred_quality"]),
                                           metadata="")
            reads_at_t.append(read)

        reads.append(reads_at_t)

    return reads


def perform_inference(reads: List[List[str]],
                      population: generative.Population,
                      time_points: List[int],
                      method: str, window_size: int, seed: int):
    random.seed(seed)

    if len(reads) != len(time_points):
        raise ValueError("There must be exactly one set of reads for each time point specified")

    if len(time_points) != len(set(time_points)):
        raise ValueError("Specified sample times must be unique")

    ##############################
    # Construct generative model
    mu = np.array([0] * len(population.strains))  # One dimension for each strain
    tau_1 = 1
    tau = 1

    my_error_model = FastQErrorModel(read_len=window_size)

    my_model = generative.GenerativeModel(times=time_points,
                                          mu=mu,
                                          tau_1=tau_1,
                                          tau=tau,
                                          bacteria_pop=population,
                                          read_length=window_size,
                                          read_error_model=my_error_model)

    if method == "em":
        solver = em.EMSolver(my_model, reads)
        abundances = solver.solve()
        logger.info("Learned abundances:")
        logger.info(abundances)

    elif method == "vi":
        posterior = vi.SecondOrderVariationalPosterior(mu, np.identity(mu.size), my_model.fragment_frequencies)
        solver = vi.SecondOrderVariationalGradientSolver(my_model, reads, posterior)
        solver.solve()

    elif method == "bbvi":
        pass  # TODO

    else:
        raise("{} is not an implemented method!".format(method))


def main():
    logger.info("Pipeline for inference started.")
    args = parse_args()
    logger.debug("Downloading marker database.")
    db = load_marker_database(args.accession_file)
    logger.debug("Creating bacteria database")
    population = parse_population(db, args.accession_file)
    logger.debug("Reading time-series read files.")
    reads = load_from_fastq(args.read_files_dir, args.read_files)
    logger.debug("Performing inference.")
    abundances = perform_inference(reads, population, args.time_points, args.method, args.read_length, args.seed)
    logger.info(str(abundances))
    logger.info("Inference finished.")


if __name__ == "__main__":
    main()

