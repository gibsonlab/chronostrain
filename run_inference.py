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
from model.reads import SequenceRead, FastQErrorModel, NoiselessErrorModel
from algs import em, bbvi, vsmc
from visualizations import plot_abundances as plotter

from typing import List
from util.io.logger import logger
from util.io.model_io import load_fastq_reads, save_abundances, load_abundances

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# ============================= Constants =================================
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

num_cores = multiprocessing.cpu_count()

_data_dir = "data"
# =========================== END Constants ===============================


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")
    # parser.add_argument('-d', '--read_files_dir', required=True,
    #                     help='<Required> Directory containing read files.')

    # Input specification.
    parser.add_argument('-r', '--read_files', nargs='+', required=True,
                        help='<Required> List of paths to read files; minimum 1.')
    parser.add_argument('-a', '--accession_path', required=True, type=str,
                        help='<Required> A path to the CSV file listing the strains to sample from. '
                             'See README for the expected format.')
    parser.add_argument('-t', '--time_points', required=True, nargs='+', type=int,
                        help='<Required> A list of integers. Each value represents a time point in the dataset.')
    parser.add_argument('-m', '--method', choices=['em', 'vi', 'bbvi'], required=True,
                        help='<Required> A keyword specifying the inference method.')

    # Output specification.
    parser.add_argument('-od', '--out_dir', required=True, type=str,
                        help='<Required> The directory to store all output files.')
    parser.add_argument('-of', '--out_file', required=True, type=str,
                        help='<Required> The filename (not the full path) to save learned outputs to.')
    parser.add_argument('-pf', '--plots_file', required=True, type=str,
                        help='<Required> The file (not the full path) to save plots to.')

    # Optional params
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-truth', '--true_abundance_path', required=False, type=str,
                        help='<Optional> The CSV file path containing the ground truth relative abundances for each '
                             'strain by time point. For benchmarking.')
    parser.add_argument('-trim', '--marker_trim_len', required=False, type=int,
                        help='<Optional> An integer to trim markers down to. For testing/debugging.')
    parser.add_argument('--disable_quality', action="store_true",
                        help='<Flag> Turn off effect of quality scores.')
    parser.add_argument('--disable_time_consistency', action="store_true",
                        help='<Flag> Turn off time consistency (perform separate inference on each time point).')

    return parser.parse_args()


def perform_em(
        reads: List[List[SequenceRead]],
        model: GenerativeModel,
        out_dir: str,
        abnd_out_filename: str,
        plot_out_filename: str,
        ground_truth_path: str,
        disable_time_consistency: bool):

    # ==== Run the solver.
    if disable_time_consistency:
        solver = em.EMSolver(model, reads, torch_device=default_device, lr=1e-5)
        abundances = solver.solve(iters=10000, print_debug_every=1000, thresh=1e-8, gradient_clip=1e5)
    else:
        def get_abundances(reads_t):
            population = model.bacteria_pop
            pseudo_model = create_model(population=population, window_size=len(reads_t[0].seq), time_points=[1])
            solver = em.EMSolver(pseudo_model, [reads_t], torch_device=default_device, lr=1e-5)
            abundances_t = solver.solve(iters=10000, print_debug_every=1000, thresh=1e-8, gradient_clip=1e5)
            return abundances_t[0]  # There are only abundances for one time point.

        model.get_fragment_space()  # Generate fragment space before running times in parallel.
        abundances = Parallel(n_jobs=num_cores)(delayed(get_abundances)(reads_t) for reads_t in tqdm(reads))
        abundances = torch.stack(abundances).to(default_device)

    # ==== Save the learned abundances.
    output_path = save_abundances(
        population=model.bacteria_pop,
        time_points=model.times,
        abundances=abundances,
        out_filename=abnd_out_filename,
        out_dir=out_dir
    )
    logger.info("Abundances saved to {}.".format(output_path))

    # ==== Get difference between learned abundances and ground-truth abundances.
    diff = None
    if ground_truth_path:
        _, predicted_abundances, _ = load_abundances(file_path=output_path, torch_device=default_device)
        _, actual_abundances, _ = load_abundances(file_path=ground_truth_path, torch_device=default_device)
        diff = torch.norm(predicted_abundances - actual_abundances, p='fro').item()
        logger.debug("Abundance squared-norm difference: {}".format(diff))

    # ==== Plot the learned abundances.
    plots_out_path = os.path.join(out_dir, plot_out_filename)
    plot_result(
        reads=reads,
        method_desc='Expectation-Maximization',
        result_path=output_path,
        true_path=ground_truth_path,
        abundance_diff=diff,
        time_consistency=time_consistency,
        plots_out_path=plots_out_path
    )
    logger.info("Plots saved to {}.".format(plots_out_path))


def perform_vsmc(
        model: GenerativeModel,
        reads: List[List[SequenceRead]]):

    # ==== Run the solver.
    solver = vsmc.VSMCSolver(model=model, data=reads, torch_device=default_device)
    solver.solve(
        optim_class=torch.optim.Adam,
        optim_args={'lr': 1e-2, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.},
        iters=1000,
        num_samples=10000,
        print_debug_every=20
    )
    posterior = solver.posterior

    # ==== Output the learned posterior.
    logger.info("Learned posterior:")
    logger.info(posterior.means[0])
    logger.info(posterior.covariances[0])

    logger.info("softmax: {}".format(posterior.means[0].data.softmax(dim=0)))
    logger.info("time 2 (AFTER SOFTMAX): {}".format(
        (posterior.means[0] + posterior.means[1]).softmax(dim=0)
    ))

    raise NotImplementedError("TODO output plots.")


def plot_result(
        reads: List[List[SequenceRead]],
        method_desc: str,
        result_path: str,
        plots_out_path: str,
        time_consistency: bool,
        abundance_diff: float = None,
        true_path: str = None):
    """
    Draw a plot of the abundances, and save to a file.

    :param reads: The collection of reads as input.
    :param method_desc: An informative name of the method used to perform inference.
    :param result_path: The path to the learned abundances.
    :param plots_out_path: The path to save the plots to.
    :param abundance_diff: The squared-norm difference between inferred abundances and ground-truth abundances.
    :param time_consistency: Whether or not the inference algorithm was performed with time-consistency.
    :param true_path: The path to the ground truth abundance file.
    (Optional. if none specified, then only plots the learned abundances.)
    :return: The path to the saved file.
    """
    num_reads_per_time = list(map(len, reads))
    avg_read_depth_over_time = sum(num_reads_per_time) / len(num_reads_per_time)

    title = "Average Read Depth over Time: " + str(round(avg_read_depth_over_time, 1)) + "\n" + \
            "Read Length: " + str(len(reads[0][0].seq)) + "\n" + \
            "Algorithm: " + method_desc + "\n" + \
            "Time consistency: " + str(time_consistency)

    if true_path:
        title += "\nSquare-Norm Abundances Difference: " + str(round(abundance_diff, 3))
        plotter.plot_abundances_comparison(
            inferred_abnd_path=result_path,
            real_abnd_path=true_path,
            title=title,
            plots_out_path=plots_out_path
        )
    else:
       plotter.plot_abundances(
            abnd_path=result_path,
            title=title,
            plots_out_path=plots_out_path
        )


def create_model(population: Population,
                 window_size: int,
                 time_points: List[int]):
    """
    Simple wrapper for creating a generative model.
    @param population: The bacteria population.
    @param window_size: Fragment read length to use.
    @param time_points: List of time points for which samples are taken from.
    @return A Generative model object.
    """
    mu = torch.zeros(len(population.strains))
    tau_1 = 100
    tau = 1
    error_model = FastQErrorModel(read_len=window_size)

    model = GenerativeModel(
        bacteria_pop=population,
        read_length=window_size,
        times=time_points,
        mu=mu,
        tau_1=tau_1,
        tau=tau,
        read_error_model=error_model,
        torch_device=default_device
    )

    return model


def main():
    logger.info("Pipeline for inference started.")
    args = parse_args()
    torch.manual_seed(args.seed)

    # ==== Create database instance.
    logger.info("Loading from marker database {}.".format(args.accession_path))
    db = SimpleCSVStrainDatabase(args.accession_path, trim_debug=args.marker_trim_len)

    # ==== Load Population instance from database info
    population = Population(
        strains=db.all_strains(),
        torch_device=default_device
    )

    # ==== Load reads and validate.
    logger.info("Loading time-series read files.")
    reads = load_fastq_reads(file_paths=args.read_files)
    logger.info("Performing inference using method '{}'.".format(args.method))

    if len(reads) != len(args.time_points):
        raise ValueError("There must be exactly one set of reads for each time point specified.")

    if len(args.time_points) != len(set(args.time_points)):
        raise ValueError("Specified sample times must be distinct.")

    """
    # ==== Create model instance.
    mu = torch.zeros(len(population.strains), device=default_device)
    tau_1 = 100
    tau = 1
    window_size = len(reads[0][0].seq)  # Assumes reads are all of the same length.

    if args.disable_quality:
        logger.info("Flag --disable_quality turned on; Quality scores are diabled.")
        error_model = NoiselessErrorModel(mismatch_likelihood=1e-100)
    else:
        error_model = FastQErrorModel(read_len=window_size)
    model = GenerativeModel(
        times=args.time_points,
        mu=mu,
        tau_1=tau_1,
        tau=tau,
        bacteria_pop=population,
        read_length=window_size,
        read_error_model=error_model,
        torch_device=default_device
    )
    """

    # ==== Create model instance
    model = create_model(population=population, window_size=len(reads[0][0].seq), time_points=args.time_points)

    logger.debug("Strain keys:")
    for k, strain in enumerate(model.bacteria_pop.strains):
        logger.debug("{} -> {}".format(strain, k))

    """
    Perform inference using the chosen method. Available choices: 'em', 'bbvi'.
    1) 'em' runs Expectation-Maximization. Saves the learned abundances and plots them.
    2) 'bbvi' runs black-box VI and saves the learned posterior parametrization (as tensors).
    """

    if args.method == 'em':
        logger.info("Solving using Expectation-Maximization (Time consistency is {})".format(args.time_consistency))
        perform_em(
            reads=reads,
            model=model,
            out_dir=args.out_dir,
            abnd_out_filename=args.out_file,
            plot_out_filename=args.plots_file,
            ground_truth_path=args.true_abundance_path,
            time_consistency=args.time_consistency
        )
    elif args.method == 'vsmc':
        logger.info("Solving using Variational Sequential Monte-Carlo.")
        perform_vsmc(
            model=model,
            reads=reads
        )
    else:
        raise ValueError("{} is not an implemented method.".format(args.method))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
