#!/bin/python3
"""
  run_inference.py
  Run to perform inference on specified reads.
"""

import argparse
from database.base import *

import torch

from model.generative import GenerativeModel
from model.bacteria import Population
from model.reads import SequenceRead, FastQErrorModel
from algs import em, vi, bbvi

from typing import List
from util.io.logger import logger
from util.io.model_io import get_all_accessions_csv, load_fastq_reads, load_abundances, save_abundances

# ============================= Constants =================================
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

_data_dir = "data"
# =========================== END Constants ===============================


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")
    parser.add_argument('-d', '--read_files_dir', required=True,
                        help='<Required> Directory containing read files.')
    parser.add_argument('-r', '--read_files', nargs='+', required=True,
                        help='<Required> One read file per time point (minimum 1)')
    parser.add_argument('-a', '--accession_file', required=True, type=str,
                        help='<Required> File listing the species to sample from. '
                             'Expected CSV format (incl. header row): '
                             '"Name","Accession" \
                              "Clostridium sporogenes ATCC 15579","NZ_DS981518.1" \
                              "Enterococcus faecalis V583","NC_004668.1" \
                              "Bacteroides fragilis NCTC9343","CR626927.1"')
    parser.add_argument('-t', '--time_points', required=True, nargs='+', type=int,
                        help='<Required> A list of integers. Each value represents a time point in the dataset.')
    parser.add_argument('-m', '--method', choices=['em', 'vi', 'bbvi'], required=True,
                        help='<Required> A keyword specifying the inference method.')
    parser.add_argument('-o', '--out_file', required=True, type=str,
                        help='The file to save results to.')

    # Optional params
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-b', '--abundance_file', required=False, type=str,
                        help='<Optional> A csv containing the relative abundances for each strain by time point.')
    parser.add_argument('-trim', '--marker_trim_len', required=False, type=int,
                        help='<Optional> An integer to trim markers down to. For testing/debugging.')

    return parser.parse_args()


def perform_inference(reads: List[List[SequenceRead]],
                      population: Population,
                      time_points: List[int],
                      method: str,
                      seed: int,
                      out_filename: str):
    torch.manual_seed(seed)

    if len(reads) != len(time_points):
        raise ValueError("There must be exactly one set of reads for each time point specified")

    if len(time_points) != len(set(time_points)):
        raise ValueError("Specified sample times must be unique")

    ##############################
    # Construct generative model
    mu = torch.zeros(len(population.strains))
    tau_1 = 100
    tau = 1
    window_size = len(reads[0][0].seq)

    my_error_model = FastQErrorModel(read_len=window_size)
    # my_error_model = NoiselessErrorModel()

    my_model = GenerativeModel(times=time_points,
                               mu=mu,
                               tau_1=tau_1,
                               tau=tau,
                               bacteria_pop=population,
                               read_length=window_size,
                               read_error_model=my_error_model,
                               torch_device=default_device)
    print(my_model.get_fragment_space())

    # logger.debug(str(my_model.get_fragment_space()))
    logger.debug("Strain keys:")
    for k, strain in enumerate(my_model.bacteria_pop.strains):
        logger.debug("{} -> {}".format(strain, k))

    if method == "em":
        logger.info("Solving using Expectation-Maximization.")
        solver = em.EMSolver(my_model, reads, torch_device=default_device, lr=1e-4)
        abundances = solver.solve(iters=10000, print_debug_every=100, thresh=1e-7, gradient_clip=1e2)
        save_abundances(
            population=population,
            time_points=time_points,
            abundances=abundances,
            out_filename=out_filename,
        )

    elif method == "vi":
        logger.info("Solving using second-order variational inference.")
        posterior = vi.SecondOrderVariationalPosterior(
            means=mu,
            covariances=torch.eye(len(population.strains)),
            frag_freqs=my_model.get_fragment_frequencies())
        solver = vi.SecondOrderVariationalGradientSolver(my_model, reads, posterior)
        return solver.solve()

    elif method == "bbvi":
        logger.info("Solving using black-box (monte-carlo) variational inference.")
        solver = bbvi.BBVISolver(model=my_model, data=reads, device=default_device)
        solver.solve()
        posterior = solver.posterior

        logger.info("Learned posterior:")
        logger.info(posterior.params())

        logger.info("Posterior sample:")
        sample_x, sample_f = posterior.sample()
        logger.info("X: ", sample_x)
        logger.info("F: ", sample_f)

    else:
        raise ValueError("{} is not an implemented method!".format(method))


def main():
    logger.info("Pipeline for inference started.")
    args = parse_args()
    logger.info("Loading from marker database {}.".format(args.accession_file))
    db = SimpleCSVStrainDatabase(args.accession_file, trim_debug=args.marker_trim_len)

    # ==== Load Population instance from database info
    accessions = get_all_accessions_csv(data_dir=_data_dir, accession_csv_file=args.accession_file)
    population = Population(
        strains=[db.get_strain(strain_id) for strain_id in accessions],
        torch_device=default_device
    )

    logger.info("Loading time-series read files.")
    reads = load_fastq_reads(base_dir=args.read_files_dir, filenames=args.read_files)
    logger.info("Performing inference using method '{}'.".format(args.method))
    predicted_abundances = perform_inference(reads, population, args.time_points, args.method, args.seed, args.out_file)
    logger.info("Inference finished.")

    if args.abundance_file:
        actual_abundances_raw = load_abundances(data_dir=_data_dir, filename=args.abundance_file)
        actual_abundances = torch.tensor(
            [[i / sum(Z) for i in Z] for Z in actual_abundances_raw],
            device=default_device
        )
        logger.info("Actual Abundances:")
        logger.info(actual_abundances)
        diff = torch.norm(predicted_abundances - actual_abundances, p='fro')
        logger.info("Difference {}".format(diff))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
