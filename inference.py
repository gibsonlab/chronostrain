"""
  inference.py
  Run to perform inference on specified reads.
  TODO: This script is still in development.
"""
import argparse
from util.logger import logger
from model import bacteria, generative
from algs import model_solver
from Bio import SeqIO
import numpy as np

from model.generative import Marker, Strain, Population


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")
    parser.add_argument('-r', '--read_files', nargs='+', required=True,
                        help='<Required> One read file per time point (minimum 1)')
    parser.add_argument('-t', '--time_points', nargs='+', help='List of time points.')
    parser.add_argument('-m', '--method', choices=['em', 'vi', 'bbvi'], required=True,
                        help='<Required> Inference method.')

    # =============
    # TODO: add more arguments to parser.
    # =============
    return parser.parse_args()


def load_marker_database():
    # TODO -- could just return a default implementation for now.
    raise NotImplementedError("TODO implement!")


def load_from_fastq(filenames):
    '''
    filesnames - a list of strings of filenames.
            filenames[0] == name of fastq file for first time point
            filenames[1] == name of fastq file for second time point
            etc...
    '''

    num_times = len(filenames)
    logger.debug("Number of time points: {}".format(num_times))


    # Parse the reads (include quality)
    reads = []  # A time-indexed list of read sets. Each entry is itself a list of reads for time t.
    for file in filenames:

        reads_at_t = []  # A list of reads at a particular time (i.e. the reads in 'file')
        for record in SeqIO.parse(file):
            read = generative.SequenceRead(seq=str(record.seq),
                                           quality=np.asanyarray(record.letter_annotations["phred_quality"]),
                                           metadata="")
            reads_at_t.append(read)

        reads.append(reads_at_t)

    return reads


def perform_inference(reads, population, times, method, window_size=50):

    if len(reads) != len(times):
        raise ValueError("There must be exactly one set of reads for each time point specified")

    if len(times) != len(set(times)):
        raise ValueError("Specified sample times must be unique")

    # Instantiate basic model instance
    mu = 0# TODO
    tau_1 = 1# TODO
    tau = 1 # TODO

    # 1) retrieve list of strains
    # 2) for each strain, retrieve list of markers

    model = generative.GenerativeModel(
        times=times,
        mu=mu,
        tau_1=tau_1,
        tau=tau,
        W=W,
        fragment_space=population.get_fragment_space(window_size=window_size),
        read_error_model=read_error_model,
        bacteria_pop=bacteria_pop
    )

    # # Generate bacteria population
    # num_strains = 4
    # num_markers = 1
    # fragment_length = 10
    # bacteria_pop = bacteria.Population(num_strains=num_strains,
    #                                       num_markers=num_markers,
    #                                       marker_length=fragment_length * 300,
    #                                       num_snps=(fragment_length * 300) // 100)
    #
    # # Generate error model
    # fragment_length = len(reads[0][0].seq)
    # read_error_model = reads.BasicErrorModel(read_len=fragment_length)
    #
    # # Construct generative model
    # mu = np.array([0] * bacteria_pop.num_strains)  # One dimension for each strain
    # tau_1 = 1
    # tau = 1
    # W = bacteria_pop.get_fragment_space(window_size=fragment_length)
    # fragment_space = bacteria_pop.get_fragment_space(window_size=fragment_length)
    #
    # model = generative.GenerativeModel(times=times,
    #                                       mu=mu,
    #                                       tau_1=tau_1,
    #                                       tau=tau,
    #                                       W=W,
    #                                       fragment_space=fragment_space,
    #                                       read_error_model=read_error_model,
    #                                       bacteria_pop=bacteria_pop)

    if method == "EM":
        means = model_solver.em_estimate(model, reads, tol=1e-10, iters=10000)
        rel_abundances = model.generate_relative_abundances(means)
        return rel_abundances
    elif method == "VI":
        return model_solver.variational_learn(model, reads, tol=1e-10)
    else:
        raise("{} is not an implemented method!".format(method))


def retrieve_markers(strain, src):
    pass


def parse_population(strain_ids):
    strains = [None for _ in strain_ids]
    # for each strain_id, create a Strain instance (but retrieve markers from somewhere
    i = 0
    for strain in strain_ids:
        markers = retrieve_markers(strain, src=None)
        strain_instance = Strain(markers)
        strains[i] = strain_instance
        i += 1
    return Population(strains)


def main():
    logger.info("Pipeline for inference started.")
    args = parse_args()
    logger.debug("Downloading marker database.")
    load_marker_database()
    logger.debug("Reading time-series read files.")
    reads = load_from_fastq(args.read_files)
    logger.debug("Performing inference.")
    population = parse_population(args.strain_ids)
    abundances = perform_inference(reads, population, args.times, args.method)
    logger.info(str(abundances))
    logger.info("Inference finished.")

if __name__ == "__main__":
    main()

