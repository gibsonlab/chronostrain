#!/bin/python3
"""
  run_inference.py
  Run to perform inference on specified reads.
"""
import csv
import os

import torch
import argparse
from tqdm import tqdm

from chronostrain import logger, cfg
from chronostrain.algs.vi import SecondOrderVariationalSolver
from chronostrain.algs import em, vsmc, bbvi, em_alt, bbvi_reparam
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.reads import SequenceRead, BasicFastQErrorModel, NoiselessErrorModel
from chronostrain.visualizations import *
from chronostrain.model.io import load_fastq_reads, save_abundances_by_path

from filter import Filter


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-b', '--reads_dir', required=True, type=str,
                        help='<Required> Directory containing read files. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')
    parser.add_argument('-m', '--method',
                        choices=['em', 'vi', 'bbvi', 'vsmc', 'emalt', 'bbvi_reparametrization'],
                        required=True,
                        help='<Required> A keyword specifying the inference method.')
    parser.add_argument('-l', '--read_length', required=True, type=int,
                        help='<Required> Length of each read')

    # Output specification.
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The file path to save learned outputs to.')

    # Optional params
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-truth', '--true_abundance_path', required=False, type=str,
                        help='<Optional> The CSV file path containing the ground truth relative abundances for each '
                             'strain by time point. For benchmarking.')
    parser.add_argument('--disable_time_consistency', action="store_true",
                        help='<Flag> Turn off time consistency (perform separate inference on each time point).')
    parser.add_argument('--skip_filter', action="store_true",
                        help='<Flag> Turn off read filtering.')
    parser.add_argument('--iters', required=False, type=int, default=10000,
                        help='<Optional> The number of iterations to run, if using EM or VI. Default: 10000')
    parser.add_argument('--num_samples', required=False, type=int, default=100,
                        help='<Optional> The number of samples to use for monte-carlo estimation '
                             '(for Variational solution).')
    parser.add_argument('-lr', '--learning_rate', required=False, type=float, default=1e-5,
                        help='<Optional> The learning rate to use for the optimizer, if using EM or VI. Default: 1e-5.')
    parser.add_argument('--abundances_file', required=False, default='abundances.out',
                        help='<Optional> Specify the filename for the learned abundances. '
                             'The file format depends on the method. '
                             'The file is saved to the output directory, specified by the -o option.')
    parser.add_argument('--plot_format', required=False, type=str, default="pdf")

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
        learning_rate: float,
        plot_format: str
):

    q_smoothing = 1e-30

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = em.EMSolver(model,
                             reads,
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
                window_size=len(reads_t[0]),
                time_points=[1],
                disable_quality=disable_quality
            )
            instance_solver = em.EMSolver(pseudo_model,
                                          [reads_t],
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
        abundances = [get_abundances(reads_t) for reads_t in tqdm(reads)]
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
        disable_quality=disable_quality,
        plot_format=plot_format
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
        learning_rate: float,
        plot_format: str
):

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = em_alt.EMAlternateSolver(model,
                                          reads,
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
        disable_quality=disable_quality,
        plot_format=plot_format
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
        cache_tag: str,
        plot_format: str
):

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = vsmc.VSMCSolver(model=model, data=reads, cache_tag=cache_tag)
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
        plots_out_path=plots_out_path,
        plot_format=plot_format
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
        cache_tag: str,
        plot_format: str
):

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = bbvi.BBVISolver(model=model, data=reads, cache_tag=cache_tag)
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
        plots_out_path=plots_out_path,
        plot_format=plot_format
    )
    logger.info("Plots saved to {}.".format(plots_out_path))


def perform_bbvi_reparametrization(
        model: GenerativeModel,
        reads: List[List[SequenceRead]],
        disable_time_consistency: bool,
        iters: int,
        out_base_dir: str,
        learning_rate: float,
        cache_tag: str):

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = bbvi_reparam.BBVIReparamSolver(
            model=model,
            data=reads,
            cache_tag=cache_tag,
            out_base_dir=out_base_dir
        )
        solver.solve(
            iters=iters,
            thresh=1e-5,
            print_debug_every=100,
            lr=learning_rate
        )
    else:
        raise NotImplementedError("Time-agnostic solver not implemented for `perform_bbvi_reparametrization`.")

    # TODO plot result.

    logger.info("BBVI Complete.")


def perform_vi(
        model: GenerativeModel,
        reads: List[List[SequenceRead]],
        disable_time_consistency: bool,
        disable_quality: bool,
        iters: int,
        num_samples: int,
        ground_truth_path: str,
        plots_out_path: str,
        cache_tag: str,
        plot_format: str
):

    # ==== Run the solver.
    if not disable_time_consistency:
        solver = SecondOrderVariationalSolver(model, reads, cache_tag)
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
        num_samples=15,
        plot_format=plot_format
    )
    logger.info("Plots saved to {}.".format(plots_out_path))


def plot_em_result(
        reads: List[List[SequenceRead]],
        result_path: str,
        plots_out_path: str,
        disable_time_consistency: bool,
        disable_quality: bool,
        plot_format: str,
        true_path: str = None):
    """
    Draw a plot of the abundances, and save to a file.

    :param reads: The collection of reads as input.
    :param result_path: The path to the learned abundances.
    :param plots_out_path: The path to save the plots to.
    :param disable_time_consistency: Whether or not the inference algorithm was performed with time-consistency.
    :param disable_quality: Whether or not quality scores were used.
    :param plot_format: The format (e.g. pdf, png) to output the plot.
    :param true_path: The path to the ground truth abundance file.
    (Optional. if none specified, then only plots the learned abundances.)
    :return: The path to the saved file.
    """
    num_reads_per_time = list(map(len, reads))
    avg_read_depth_over_time = sum(num_reads_per_time) / len(num_reads_per_time)

    title = "Average Read Depth over Time: " + str(round(avg_read_depth_over_time, 1)) + "\n" + \
            "Read Length: " + str(len(reads[0][0])) + "\n" + \
            "Algorithm: Expectation-Maximization" + "\n" + \
            ('Time consistency off\n' if disable_time_consistency else '') + \
            ('Quality score off\n' if disable_quality else '')

    if true_path:
        plot_abundances_comparison(
            inferred_abnd_path=result_path,
            real_abnd_path=true_path,
            title=title,
            plots_out_path=plots_out_path,
            draw_legend=False,
            img_format=plot_format
        )
    else:
        plot_abundances(
            abnd_path=result_path,
            title=title,
            plots_out_path=plots_out_path,
            draw_legend=False,
            img_format=plot_format
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
        plot_format: str,
        num_samples: int = 10000,
        truth_path: str = None):
    num_reads_per_time = list(map(len, reads))
    avg_read_depth_over_time = sum(num_reads_per_time) / len(num_reads_per_time)

    title = "Average Read Depth over Time: " + str(round(avg_read_depth_over_time, 1)) + "\n" + \
            "Read Length: " + str(len(reads[0][0])) + "\n" + \
            "Algorithm: " + method + "\n" + \
            ('Time consistency off\n' if disable_time_consistency else '') + \
            ('Quality score off\n' if disable_quality else '')

    plot_posterior_abundances(
        times=times,
        posterior=posterior,
        population=population,
        title=title,
        plots_out_path=plots_out_path,
        truth_path=truth_path,
        num_samples=num_samples,
        draw_legend=False,
        img_format=plot_format
    )


def create_model(population: Population,
                 window_size: int,
                 time_points: List[float],
                 disable_quality: bool):
    """
    Simple wrapper for creating a generative model.
    @param population: The bacteria population.
    @param window_size: Fragment read length to use.
    @param time_points: List of time points for which samples are taken from.
    @param disable_quality: A flag to indicate whether or not to use NoiselessErrorModel.
    @return A Generative model object.
    """
    mu = torch.zeros(len(population.strains), device=cfg.torch_cfg.device)
    tau_1 = 1
    tau = 1

    if disable_quality:
        logger.info("Flag --disable_quality turned on; Quality scores are diabled. Initializing NoiselessErrorModel.")
        error_model = NoiselessErrorModel(mismatch_likelihood=0.)
    else:
        error_model = BasicFastQErrorModel(read_len=window_size)

    model = GenerativeModel(
        bacteria_pop=population,
        read_length=window_size,
        times=time_points,
        mu=mu,
        tau_1=tau_1,
        tau=tau,
        read_error_model=error_model
    )

    return model


def get_input_paths(base_dir) -> Tuple[List[str], List[float]]:
    time_points = []
    read_files = []

    input_specification_path = os.path.join(base_dir, "input_files.csv")
    try:
        with open(input_specification_path, "r") as f:
            input_specs = csv.reader(f, delimiter=',', quotechar='"')
            for item in input_specs:
                time_points.append(float(item[0]))
                read_files.append(os.path.join(base_dir, item[1]))
    except FileNotFoundError:
        raise FileNotFoundError("Missing required file `input_files.csv` in directory {}.".format(base_dir)) from None

    return read_files, time_points


def main():
    logger.info("Pipeline for inference started.")
    args = parse_args()
    torch.manual_seed(args.seed)

    # ==== Create database instance.
    db = cfg.database_cfg.get_database()

    # ==== Load Population instance from database info
    population = Population(
        strains=db.all_strains()
    )

    read_paths, time_points = get_input_paths(args.reads_dir)

    # ==== Load reads and validate.
    if len(read_paths) != len(time_points):
        raise ValueError("There must be exactly one set of reads for each time point specified.")

    if len(time_points) != len(set(time_points)):
        raise ValueError("Specified sample times must be distinct.")

    if not args.skip_filter:
        logger.info("Performing filter on reads.")
        filt = Filter(
            reference_file_paths=[strain.metadata.file_path for strain in population.strains],
            reads_paths=read_paths,
            time_points=time_points,
            align_cmd=cfg.filter_cfg.align_cmd
        )
        filtered_read_files = filt.apply_filter(args.read_length)
        logger.info("Loading filtered time-series read files.")
        reads = load_fastq_reads(file_paths=filtered_read_files)
    else:
        logger.info("Loading time-series read files.")
        reads = load_fastq_reads(file_paths=read_paths)

    logger.info("Performing inference using method '{}'.".format(args.method))

    # ==== Create model instance
    model = create_model(
        population=population,
        window_size=len(reads[0][0]),
        time_points=time_points,
        disable_quality=not cfg.model_cfg.use_quality_scores
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
        ''.join(read_paths)
    )

    if args.method == 'em':
        logger.info("Solving using Expectation-Maximization.")
        out_path = os.path.join(args.out_dir, args.abundances_file)
        plots_path = os.path.join(args.out_dir, "plot.{}".format(args.plot_format))
        perform_em(
            reads=reads,
            model=model,
            abnd_out_path=out_path,
            plots_out_path=plots_path,
            ground_truth_path=args.true_abundance_path,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            iters=args.iters,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag,
            plot_format=args.plot_format
        )
    elif args.method == 'bbvi':
        logger.info("Solving using Black-Box Variational Inference.")
        plots_path = os.path.join(args.out_dir, "plot.{}".format(args.plot_format))
        perform_bbvi(
            model=model,
            reads=reads,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            iters=args.iters,
            num_samples=args.num_samples,
            ground_truth_path=args.true_abundance_path,
            plots_out_path=plots_path,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag,
            plot_format=args.plot_format
        )
    elif args.method == 'bbvi_reparametrization':
        logger.info("Solving using Black-Box Variational Inference.")
        perform_bbvi_reparametrization(
            model=model,
            reads=reads,
            disable_time_consistency=args.disable_time_consistency,
            iters=args.iters,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag,
            out_base_dir=args.out_dir
        )
    elif args.method == 'vsmc':
        logger.info("Solving using Variational Sequential Monte-Carlo.")
        plots_path = os.path.join(args.out_dir, "plot.{}".format(args.plot_format))
        perform_vsmc(
            model=model,
            reads=reads,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            iters=args.iters,
            num_samples=args.num_samples,
            ground_truth_path=args.true_abundance_path,
            plots_out_path=plots_path,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag,
            plot_format=args.plot_format
        )
    elif args.method == 'vi':
        logger.info("Solving using Variational Inference (Second-order mean-field solution).")
        plots_path = os.path.join(args.out_dir, "plot.{}".format(args.plot_format))
        perform_vi(
            model=model,
            reads=reads,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            iters=args.iters,
            num_samples=args.num_samples,
            ground_truth_path=args.true_abundance_path,
            plots_out_path=plots_path,
            cache_tag=cache_tag,
            plot_format=args.plot_format
        )
    elif args.method == 'emalt':
        out_path = os.path.join(args.out_dir, "abundances.csv")
        plots_path = os.path.join(args.out_dir, "plot.{}".format(args.plot_format))
        logger.info("Solving using Alt-EM.")
        perform_em_alt(
            reads=reads,
            model=model,
            abnd_out_path=out_path,
            plots_out_path=plots_path,
            ground_truth_path=args.true_abundance_path,
            disable_time_consistency=args.disable_time_consistency,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            iters=args.iters,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag,
            plot_format=args.plot_format
        )
    else:
        raise ValueError("{} is not an implemented method.".format(args.method))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        exit(1)
