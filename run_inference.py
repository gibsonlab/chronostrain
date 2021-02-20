#!/bin/python3
"""
  run_inference.py
  Run to perform inference on specified reads.
"""

import argparse

from algs.vi import SecondOrderVariationalSolver, AbstractVariationalPosterior
from database import JSONStrainDatabase, SimpleCSVStrainDatabase

import torch

from filter import Filter
from model.generative import GenerativeModel
from model.bacteria import Population
from model.reads import SequenceRead, FastQErrorModel, NoiselessErrorModel
from algs import em, vsmc, bbvi, em_alt
from visualizations import plot_abundances as plotter

from typing import List
from util.io.logger import logger
from util.io.model_io import load_fastq_reads, load_abundances, save_abundances_by_path

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# ============================= Constants =================================
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# default_device = torch.device("cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

num_cores = multiprocessing.cpu_count()

_data_dir = "data"
# =========================== END Constants ===============================


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-b', '--base_path', required=True, type=str,
                        help='<Required> Directory containing read files')
    parser.add_argument('-r', '--read_files', nargs='+', required=True,
                        help='<Required> List of read filenames; minimum 1.')
    parser.add_argument('-a', '--accession_path', required=True, type=str,
                        help='<Required> A path to the CSV file listing the strains to sample from. '
                             'See README for the expected format.')
    parser.add_argument('-t', '--time_points', required=True, nargs='+', type=float,
                        help='<Required> A list of integers. Each value represents a time point in the dataset.')
    parser.add_argument('-m', '--method', choices=['em', 'vi', 'bbvi', 'vsmc', 'emalt'], required=True,
                        help='<Required> A keyword specifying the inference method.')
    parser.add_argument('-l', '--read_length', required=True, type=int,
                        help='<Required> Length of each read')

    # Output specification.
    parser.add_argument('-of', '--out_path', required=True, type=str,
                        help='<Required> The file path to save learned outputs to.')
    parser.add_argument('-pf', '--plots_path', required=True, type=str,
                        help='<Required> The file path to save plots to.')

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
    parser.add_argument('--iters', required=False, type=int, default=10000,
                        help='<Optional> The number of iterations to run, if using EM or VI. Default: 10000')
    parser.add_argument('--num_samples', required=False, type=int, default=100,
                        help='<Optional> The number of samples to use for monte-carlo estimation '
                             '(for Variational solution).')
    parser.add_argument('-lr', '--learning_rate', required=False, type=float, default=1e-5,
                        help='<Optional> The learning rate to use for the optimizer, if using EM or VI. Default: 1e-5.')

    return parser.parse_args()


def perform_em(
        reads: List[List[SequenceRead]],
        model: GenerativeModel,
        abnd_out_path: str,
        plots_out_path: str,
        ground_truth_path: str,
        disable_time_consistency: bool,
        disable_quality: bool,
        iters: int,
        cache_tag: str,
        learning_rate: float):

    q_smoothing = 1e-30

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = em.EMSolver(model,
                             reads,
                             device=default_device,
                             cache_tag=cache_tag,
                             lr=learning_rate)
        abundances = solver.solve(
            iters=iters,
            print_debug_every=1000,
            thresh=1e-5,
            gradient_clip=1e5,
            q_smoothing=q_smoothing)
    else:
        logger.info("Flag --disable_time_consistency turned on; Performing inference on each sample independently.")

        def get_abundances(reads_t):
            population = model.bacteria_pop
            pseudo_model = create_model(
                population=population,
                window_size=len(reads_t[0].seq),
                time_points=[1],
                disable_quality=disable_quality
            )
            instance_solver = em.EMSolver(pseudo_model,
                                          [reads_t],
                                          device=default_device,
                                          cache_tag=cache_tag,
                                          lr=learning_rate)
            abundances_t = instance_solver.solve(
                iters=10000,
                print_debug_every=1000,
                thresh=1e-5,
                gradient_clip=1e5,
                q_smoothing=q_smoothing)
            return abundances_t[0]  # There are only abundances for one time point.

        # Generate fragment space (stored and shared in Population instance) before running times in parallel.
        model.get_fragment_space()

        # Run jobs distributed across processes.
        abundances = Parallel(n_jobs=num_cores)(delayed(get_abundances)(reads_t) for reads_t in tqdm(reads))
        abundances = torch.stack(abundances)

    # ==== Save the learned abundances.
    output_path = save_abundances_by_path(
        population=model.bacteria_pop,
        time_points=model.times,
        abundances=abundances,
        out_path=abnd_out_path
    )
    logger.info("Abundances saved to {}.".format(output_path))

    # ==== Plot the learned abundances.
    logger.info("Done. Saving plot of learned abundances.")
    plot_em_result(
        reads=reads,
        result_path=output_path,
        true_path=ground_truth_path,
        plots_out_path=plots_out_path,
        disable_time_consistency=disable_time_consistency,
        disable_quality=disable_quality
    )
    logger.info("Plots saved to {}.".format(plots_out_path))


def perform_em_alt(
        reads: List[List[SequenceRead]],
        model: GenerativeModel,
        abnd_out_path: str,
        plots_out_path: str,
        ground_truth_path: str,
        disable_time_consistency: bool,
        disable_quality: bool,
        iters: int,
        cache_tag: str,
        learning_rate: float):

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = em_alt.EMAlternateSolver(model,
                                          reads,
                                          device=default_device,
                                          cache_tag=cache_tag,
                                          lr=learning_rate)
        abundances, strains = solver.solve(
            max_iters=iters,
            print_debug_every=1,
            x_opt_thresh=1e-5
        )
    else:
        raise NotImplementedError()

    for t in range(len(reads)):
        for read, strain in zip(reads[t], strains[t]):
            logger.debug("{} -> {}".format(read.metadata, strain))

    # ==== Save the learned abundances.
    output_path = save_abundances_by_path(
        population=model.bacteria_pop,
        time_points=model.times,
        abundances=abundances,
        out_path=abnd_out_path
    )
    logger.info("Abundances saved to {}.".format(output_path))

    # ==== Plot the learned abundances.
    logger.info("Done. Saving plot of learned abundances.")
    plot_em_result(
        reads=reads,
        result_path=output_path,
        true_path=ground_truth_path,
        plots_out_path=plots_out_path,
        disable_time_consistency=disable_time_consistency,
        disable_quality=disable_quality
    )
    logger.info("Plots saved to {}.".format(plots_out_path))


def perform_vsmc(
        model: GenerativeModel,
        reads: List[List[SequenceRead]],
        disable_time_consistency: bool,
        disable_quality: bool,
        iters: int,
        learning_rate: float,
        num_samples: int,
        ground_truth_path: str,
        plots_out_path: str,
        cache_tag: str):

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = vsmc.VSMCSolver(model=model, data=reads, torch_device=default_device, cache_tag=cache_tag)
        solver.solve(
            optim_class=torch.optim.Adam,
            optim_args={'lr': learning_rate, 'betas': (0.7, 0.7), 'eps': 1e-7, 'weight_decay': 0.},
            iters=iters,
            num_samples=num_samples,
            print_debug_every=100
        )
        posterior = solver.posterior
    else:
        raise NotImplementedError("Feature 'disable_time_consistency' not implemented for VSMC.")

    logger.info("Done. Generating plot of posterior.")
    plot_variational_result(
        method='Variational Sequential Monte Carlo',
        times=model.times,
        population=model.bacteria_pop,
        reads=reads,
        posterior=posterior,
        disable_time_consistency=disable_time_consistency,
        disable_quality=disable_quality,
        truth_path=ground_truth_path,
        plots_out_path=plots_out_path
    )
    logger.info("Plots saved to {}.".format(plots_out_path))


def perform_bbvi(
        model: GenerativeModel,
        reads: List[List[SequenceRead]],
        disable_time_consistency: bool,
        disable_quality: bool,
        iters: int,
        learning_rate: float,
        num_samples: int,
        ground_truth_path: str,
        plots_out_path: str,
        cache_tag: str):

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = bbvi.BBVISolver(model=model, data=reads, device=default_device, cache_tag=cache_tag)
        solver.solve(
            optim_class=torch.optim.Adam,
            optim_args={'lr': learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.},
            iters=iters,
            num_samples=num_samples,
            print_debug_every=100
        )
        posterior = solver.posterior
    else:
        raise NotImplementedError("Feature 'disable_time_consistency' not implemented for BBVI.")

    logger.info("Done. Generating plot of posterior.")
    plot_variational_result(
        method='Black-Box Variational Inference',
        times=model.times,
        population=model.bacteria_pop,
        reads=reads,
        posterior=posterior,
        disable_time_consistency=disable_time_consistency,
        disable_quality=disable_quality,
        truth_path=ground_truth_path,
        plots_out_path=plots_out_path
    )
    logger.info("Plots saved to {}.".format(plots_out_path))


def perform_vi(
        model: GenerativeModel,
        reads: List[List[SequenceRead]],
        disable_time_consistency: bool,
        disable_quality: bool,
        iters: int,
        num_samples: int,
        ground_truth_path: str,
        plots_out_path: str,
        cache_tag: str):

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = SecondOrderVariationalSolver(model, reads, default_device, cache_tag)
        posterior = solver.solve(
            iters=iters,
            num_montecarlo_samples=num_samples,
            print_debug_every=1,
            thresh=1e-10,
            clipping=0.3,
            stdev_scale=[50, 50, 50, 50, 50, 300, 50, 500]
        )
    else:
        raise NotImplementedError("Feature 'disable_time_consistency' not implemented for VI.")

    logger.info("Done. Generating plot of posterior.")
    plot_variational_result(
        method='Variational Inference (Second-order heuristic)',
        times=model.times,
        population=model.bacteria_pop,
        reads=reads,
        posterior=posterior,
        disable_time_consistency=disable_time_consistency,
        disable_quality=disable_quality,
        truth_path=ground_truth_path,
        plots_out_path=plots_out_path,
        num_samples=15
    )
    logger.info("Plots saved to {}.".format(plots_out_path))


def plot_em_result(
        reads: List[List[SequenceRead]],
        result_path: str,
        plots_out_path: str,
        disable_time_consistency: bool,
        disable_quality: bool,
        true_path: str = None):
    """
    Draw a plot of the abundances, and save to a file.

    :param reads: The collection of reads as input.
    :param result_path: The path to the learned abundances.
    :param plots_out_path: The path to save the plots to.
    :param disable_time_consistency: Whether or not the inference algorithm was performed with time-consistency.
    :param disable_quality: Whether or not quality scores were used.
    :param true_path: The path to the ground truth abundance file.
    (Optional. if none specified, then only plots the learned abundances.)
    :return: The path to the saved file.
    """
    num_reads_per_time = list(map(len, reads))
    avg_read_depth_over_time = sum(num_reads_per_time) / len(num_reads_per_time)

    title = "Average Read Depth over Time: " + str(round(avg_read_depth_over_time, 1)) + "\n" + \
            "Read Length: " + str(len(reads[0][0].seq)) + "\n" + \
            "Algorithm: Expectation-Maximization" + "\n" + \
            ('Time consistency off\n' if disable_time_consistency else '') + \
            ('Quality score off\n' if disable_quality else '')

    if true_path:
        # title += "\nSquare-Norm Abundances Difference: " + str(round(abundance_diff, 3))
        plotter.plot_abundances_comparison(
            inferred_abnd_path=result_path,
            real_abnd_path=true_path,
            title=title,
            plots_out_path=plots_out_path,
            draw_legend=False
        )
    else:
        plotter.plot_abundances(
            abnd_path=result_path,
            title=title,
            plots_out_path=plots_out_path,
            draw_legend=False
        )


def plot_variational_result(
        method: str,
        times: List[float],
        population: Population,
        reads: List[List[SequenceRead]],
        posterior: AbstractVariationalPosterior,
        disable_time_consistency: bool,
        disable_quality: bool,
        plots_out_path: str,
        num_samples: int = 10000,
        truth_path: str = None):
    num_reads_per_time = list(map(len, reads))
    avg_read_depth_over_time = sum(num_reads_per_time) / len(num_reads_per_time)

    title = "Average Read Depth over Time: " + str(round(avg_read_depth_over_time, 1)) + "\n" + \
            "Read Length: " + str(len(reads[0][0].seq)) + "\n" + \
            "Algorithm: " + method + "\n" + \
            ('Time consistency off\n' if disable_time_consistency else '') + \
            ('Quality score off\n' if disable_quality else '')

    plotter.plot_posterior_abundances(
        times=times,
        posterior=posterior,
        population=population,
        title=title,
        plots_out_path=plots_out_path,
        truth_path=truth_path,
        num_samples=num_samples,
        draw_legend=False
    )


def create_model(population: Population,
                 window_size: int,
                 time_points: List[int],
                 disable_quality: bool):
    """
    Simple wrapper for creating a generative model.
    @param population: The bacteria population.
    @param window_size: Fragment read length to use.
    @param time_points: List of time points for which samples are taken from.
    @param disable_quality: A flag to indicate whether or not to use NoiselessErrorModel.
    @return A Generative model object.
    """
    mu = torch.zeros(len(population.strains), device=default_device)
    tau_1 = 1
    tau = 1

    if disable_quality:
        logger.info("Flag --disable_quality turned on; Quality scores are diabled. Initializing NoiselessErrorModel.")
        error_model = NoiselessErrorModel(mismatch_likelihood=0.)
    else:
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
    if args.accession_path.endswith(".csv"):
        db = SimpleCSVStrainDatabase(args.accession_path, trim_debug=args.marker_trim_len)
    else:
        db = JSONStrainDatabase(args.accession_path)
    # ==== Load Population instance from database info
    population = Population(
        strains=db.all_strains(),
        torch_device=default_device
    )

    # ==== Load reads and validate.
    if len(args.read_files) != len(args.time_points):
        raise ValueError("There must be exactly one set of reads for each time point specified.")

    if len(args.time_points) != len(set(args.time_points)):
        raise ValueError("Specified sample times must be distinct.")

    filter = Filter(db.dump_markers_to_fasta(args.base_path), args.base_path, args.read_files, args.time_points)
    filtered_read_files = filter.apply_filter(args.read_length)

    logger.info("Loading time-series read files.")
    reads = load_fastq_reads(file_paths=filtered_read_files)
    logger.info("Performing inference using method '{}'.".format(args.method))

    # ==== Create model instance
    model = create_model(
        population=population,
        window_size=len(reads[0][0].seq),
        time_points=args.time_points,
        disable_quality=args.disable_quality
    )

    logger.debug("Strain keys:")
    for k, strain in enumerate(model.bacteria_pop.strains):
        logger.debug("{} -> {}".format(strain, k))

    """
    Perform inference using the chosen method. Available choices: 'em', 'bbvi'.
    1) 'em' runs Expectation-Maximization. Saves the learned abundances and plots them.
    2) 'bbvi' runs black-box VI and saves the learned posterior parametrization (as tensors).
    """
    cache_tag = "{}_{}".format(
        args.method,
        ''.join(args.read_files)
    )

    if args.method == 'em':
        logger.info("Solving using Expectation-Maximization.")
        perform_em(
            reads=reads,
            model=model,
            abnd_out_path=args.out_path,
            plots_out_path=args.plots_path,
            ground_truth_path=args.true_abundance_path,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=args.disable_quality,
            iters=args.iters,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag
        )
    elif args.method == 'bbvi':
        logger.info("Solving using Black-Box Variational Inference.")
        perform_bbvi(
            model=model,
            reads=reads,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=args.disable_quality,
            iters=args.iters,
            num_samples=args.num_samples,
            ground_truth_path=args.true_abundance_path,
            plots_out_path=args.plots_path,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag
        )
    elif args.method == 'vsmc':
        logger.info("Solving using Variational Sequential Monte-Carlo.")
        perform_vsmc(
            model=model,
            reads=reads,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=args.disable_quality,
            iters=args.iters,
            num_samples=args.num_samples,
            ground_truth_path=args.true_abundance_path,
            plots_out_path=args.plots_path,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag
        )
    elif args.method == 'vi':
        logger.info("Solving using Variational Inference (Second-order mean-field solution).")
        perform_vi(
            model=model,
            reads=reads,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=args.disable_quality,
            iters=args.iters,
            num_samples=args.num_samples,
            ground_truth_path=args.true_abundance_path,
            plots_out_path=args.plots_path,
            cache_tag=cache_tag
        )
    elif args.method == 'emalt':
        logger.info("Solving using Alt-EM.")
        perform_em_alt(
            reads=reads,
            model=model,
            abnd_out_path=args.out_path,
            plots_out_path=args.plots_path,
            ground_truth_path=args.true_abundance_path,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=args.disable_quality,
            iters=args.iters,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag
        )
    else:
        raise ValueError("{} is not an implemented method.".format(args.method))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        exit(1)
