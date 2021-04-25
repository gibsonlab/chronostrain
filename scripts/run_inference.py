"""
  run_inference.py
  Run to perform inference on specified reads.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
import argparse

from matplotlib import pyplot as plt
from matplotlib import animation
from typing import Optional, List, Tuple

from chronostrain import logger, cfg
from chronostrain.algs import AbstractPosterior, BBVISolver, EMSolver
from chronostrain.model import Population
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.reads import BasicFastQErrorModel, NoiselessErrorModel
from chronostrain.model.io import TimeSeriesReads, save_abundances
from chronostrain.util.data_cache import CacheTag
from chronostrain.util import filesystem
from chronostrain.visualizations import plot_posterior_abundances, plot_abundances_comparison, plot_abundances


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-r', '--reads_dir', required=True, type=str,
                        help='<Required> Directory containing read files. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')
    parser.add_argument('-m', '--method',
                        choices=['em', 'bbvi', 'bbvi_reparametrization'],
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
    parser.add_argument('--num_posterior_samples', required=False, type=int, default=5000,
                        help='<Optional> If using a variational method, specify the number of '
                             'samples to generate as output.')
    parser.add_argument('--plot_format', required=False, type=str, default="pdf")

    return parser.parse_args()


def perform_em(
        reads: TimeSeriesReads,
        model: GenerativeModel,
        out_dir: Path,
        abnd_out_file: str,
        ground_truth_path: Path,
        disable_quality: bool,
        iters: int,
        cache_tag: CacheTag,
        learning_rate: float,
        plot_format: str
):

    q_smoothing = 1e-30

    # ==== Run the solver.
    solver = EMSolver(model,
                         reads,
                         cache_tag=cache_tag,
                         lr=learning_rate)
    abundances, var_1, var = solver.solve(
        iters=iters,
        print_debug_every=1000,
        thresh=1e-5,
        gradient_clip=1e5,
        q_smoothing=q_smoothing
    )

    # ==== Save the learned abundances.
    output_path = out_dir / abnd_out_file
    save_abundances(
        population=model.bacteria_pop,
        time_points=model.times,
        abundances=abundances,
        out_path=output_path
    )
    logger.info("Abundances saved to {}.".format(output_path))

    metadata_path = out_dir / "metadata.txt"
    with open(metadata_path, "w") as metadata_file:
        print("Learned var_1: {}".format(var_1), file=metadata_file)
        print("Learned var: {}".format(var), file=metadata_file)
    logger.info("Wrote metadata to {}.".format(metadata_path))

    # ==== Plot the learned abundances.
    logger.info("Done. Saving plot of learned abundances.")
    plot_path = out_dir / "plot.{}".format(plot_format)
    plot_em_result(
        reads=reads,
        result_path=output_path,
        true_path=ground_truth_path,
        plots_out_path=plot_path,
        disable_quality=disable_quality,
        plot_format=plot_format
    )
    logger.info("Plots saved to {}.".format(plot_path))


# def perform_vsmc(
#         model: GenerativeModel,
#         reads: TimeSeriesReads,
#         disable_quality: bool,
#         iters: int,
#         learning_rate: float,
#         num_samples: int,
#         ground_truth_path: Path,
#         plots_out_path: Path,
#         samples_out_path: Path,
#         cache_tag: CacheTag,
#         plot_format: str
# ):
#
#     # ==== Run the solver.
#     solver = vsmc.VSMCSolver(model=model, data=reads, cache_tag=cache_tag)
#     solver.solve(
#         optim_class=torch.optim.Adam,
#         optim_args={'lr': learning_rate, 'betas': (0.7, 0.7), 'eps': 1e-7, 'weight_decay': 0.},
#         iters=iters,
#         num_samples=num_samples,
#         print_debug_every=100
#     )
#     posterior = solver.posterior
#
#     logger.info("Done. Generating plot of posterior.")
#     output_variational_result(
#         method='Variational Sequential Monte Carlo',
#         model=model,
#         posterior=posterior,
#         disable_quality=disable_quality,
#         truth_path=ground_truth_path,
#         plots_out_path=plots_out_path,
#         samples_out_path=samples_out_path,
#         plot_format=plot_format
#     )
#     logger.info("Plots saved to {}.".format(plots_out_path))


def perform_bbvi(
        model: GenerativeModel,
        reads: TimeSeriesReads,
        disable_quality: bool,
        iters: int,
        learning_rate: float,
        num_samples: int,
        ground_truth_path: Path,
        out_dir: Path,
        cache_tag: CacheTag,
        plot_format: str,
        plot_elbo_history: bool = True,
        do_training_animation: bool = False,
        correlation_type: str = "full"
):

    # ==== Run the solver.
    solver = BBVISolver(model=model, data=reads, cache_tag=cache_tag, correlation_type=correlation_type)

    callbacks = []

    uppers = [[] for _ in range(model.num_strains())]
    lowers = [[] for _ in range(model.num_strains())]
    medians = [[] for _ in range(model.num_strains())]
    elbo_history = []

    if do_training_animation:
        def anim_callback(iter, x_samples, elbo):
            # Plot BBVI posterior.
            if iter % 20 != 0:
                return
            abund_samples = softmax(x_samples, dim=2).cpu().detach().numpy()
            for s_idx in range(model.num_strains()):
                traj_samples = abund_samples[:, :, s_idx]  # (T x N)
                upper_quantile = np.quantile(traj_samples, q=0.975, axis=1)
                lower_quantile = np.quantile(traj_samples, q=0.025, axis=1)
                median = np.quantile(traj_samples, q=0.5, axis=1)
                uppers[s_idx].append(upper_quantile)
                lowers[s_idx].append(lower_quantile)
                medians[s_idx].append(median)

        callbacks.append(anim_callback)

    if plot_elbo_history:
        def elbo_callback(iter, x_samples, elbo):
            elbo_history.append(elbo)
        callbacks.append(elbo_callback)

    solver.solve(
        optim_class=torch.optim.Adam,
        optim_args={'lr': learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.},
        iters=iters,
        num_samples=num_samples,
        print_debug_every=100,
        callbacks=callbacks
    )
    posterior = solver.gaussian_posterior

    if plot_elbo_history:
        elbo_plot_path = out_dir / "elbo.{}".format(plot_format)
        plot_elbos(out_path=elbo_plot_path, elbos=elbo_history, plot_format=plot_format)
        logger.info("Saved ELBO plot to {}.".format(elbo_plot_path))

    if do_training_animation:
        animation_plot_path = out_dir / "training.gif"
        plot_training_animation(
            out_path=animation_plot_path,
            n_frames=len(uppers[0]),
            lowers=lowers,
            uppers=uppers,
            medians=medians,
            model=model
        )
        logger.info("Saved training history to {}.".format(animation_plot_path))

    # ==== Save the fragment probabilities.
    df_entries = []
    for t_idx, reads_t in enumerate(reads):
        for solver_r_idx, r_idx in enumerate(solver.read_indices[t_idx]):
            read = reads_t[r_idx]
            for fragment, frag_prob in solver.fragment_posterior.top_fragments(t_idx, solver_r_idx, top=5):
                if frag_prob < 0.05:
                    continue
                df_entries.append({
                    "time_idx": t_idx,
                    "read_idx": r_idx,
                    "read_id": read.id,
                    "frag_seq": fragment.nucleotide_content(),
                    "frag_prob": frag_prob,
                    "frag_metadata": fragment.metadata,
                })
    df = pd.DataFrame(df_entries).set_index(["time_idx", "read_idx"])

    frag_prob_path = out_dir / "reads_to_frags.csv"
    df.to_csv(str(frag_prob_path), mode="w")
    logger.info("Saved read-to-fragment likelihoods to {} [{}].".format(
        frag_prob_path,
        filesystem.convert_size(frag_prob_path.stat().st_size)
    ))

    # ==== Plot the posterior.
    logger.info("Done. Generating plot of posterior.")
    plot_out_path = out_dir / "plot.{}".format(plot_format)
    output_variational_result(
        method='Black-Box Variational Inference',
        model=model,
        posterior=posterior,
        disable_quality=disable_quality,
        truth_path=ground_truth_path,
        plots_out_path=plot_out_path,
        samples_out_path=out_dir / "samples.pt",
        plot_format=plot_format
    )
    logger.info("Plots saved to {}.".format(plot_out_path))


# def perform_bbvi_reparametrization(
#         model: GenerativeModel,
#         reads: TimeSeriesReads,
#         disable_quality: bool,
#         iters: int,
#         learning_rate: float,
#         cache_tag: CacheTag,
#         plot_out_path: Path,
#         samples_path: Path,
#         plot_format: str,
#         ground_truth_path: Path = None,
#         num_posterior_samples: int = 5000):
#
#     # ==== Run the solver.
#     solver = BBVIReparamSolver(
#         model=model,
#         data=reads,
#         cache_tag=cache_tag,
#     )
#     bbvi_posterior = solver.solve(
#         iters=iters,
#         thresh=1e-5,
#         print_debug_every=1,
#         lr=learning_rate
#     )
#
#     torch.save(bbvi_posterior.sample(num_samples=num_posterior_samples), samples_path)
#     logger.info("Posterior samples saved to {}. [{}]".format(
#         samples_path,
#         filesystem.convert_size(samples_path.stat().st_size)
#     ))
#
#     output_variational_result(
#         method="Black Box Variational Inference (with reparametrization)",
#         model=model,
#         posterior=bbvi_posterior,
#         disable_quality=disable_quality,
#         plots_out_path=plot_out_path,
#         samples_out_path=samples_path,
#         plot_format=plot_format,
#         num_samples=num_posterior_samples,
#         truth_path=ground_truth_path
#     )
#     logger.info("Plots saved to {}.".format(plot_out_path))


# def perform_vi(
#         model: GenerativeModel,
#         reads: TimeSeriesReads,
#         disable_quality: bool,
#         iters: int,
#         num_samples: int,
#         ground_truth_path: Path,
#         plots_out_path: Path,
#         samples_out_path: Path,
#         cache_tag: CacheTag,
#         plot_format: str
# ):
#
#     # ==== Run the solver.
#     solver = SecondOrderVariationalSolver(model, reads, cache_tag)
#     posterior = solver.solve(
#         iters=iters,
#         num_montecarlo_samples=num_samples,
#         print_debug_every=1,
#         thresh=1e-10,
#         clipping=0.3,
#         stdev_scale=[50, 50, 50, 50, 50, 300, 50, 500]
#     )
#
#     logger.info("Done. Generating plot of posterior.")
#     output_variational_result(
#         method='Variational Inference (Second-order heuristic)',
#         model=model,
#         posterior=posterior,
#         disable_quality=disable_quality,
#         truth_path=ground_truth_path,
#         plots_out_path=plots_out_path,
#         samples_out_path=samples_out_path,
#         num_samples=15,
#         plot_format=plot_format
#     )
#     logger.info("Plots saved to {}.".format(plots_out_path))


def plot_em_result(
        reads: TimeSeriesReads,
        result_path: Path,
        plots_out_path: Path,
        disable_quality: bool,
        plot_format: str,
        true_path: Optional[Path] = None):
    """
    Draw a plot of the abundances, and save to a file.

    :param reads: The collection of reads as input.
    :param result_path: The path to the learned abundances.
    :param plots_out_path: The path to save the plots to.
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
            ('Quality score off\n' if disable_quality else '')

    if true_path is not None:
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


def output_variational_result(
        method: str,
        model: GenerativeModel,
        posterior: AbstractPosterior,
        disable_quality: bool,
        plots_out_path: Path,
        samples_out_path: Path,
        plot_format: str,
        num_samples: int = 10000,
        truth_path: Optional[Path] = None):
    # Samples.
    samples = posterior.sample(num_samples)
    torch.save(samples, samples_out_path)
    logger.info("Posterior samples saved to {}. [{}]".format(
        samples_out_path,
        filesystem.convert_size(samples_out_path.stat().st_size)
    ))

    # Plotting.
    title = "Algorithm: " + method + "\n" + \
            ('Quality score off\n' if disable_quality else '')

    plot_posterior_abundances(
        times=model.times,
        posterior_samples=samples.cpu().numpy(),
        population=model.bacteria_pop,
        title=title,
        plots_out_path=plots_out_path,
        truth_path=truth_path,
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
        tau_1_dof=cfg.model_cfg.sics_dof_1,
        tau_1_scale=cfg.model_cfg.sics_scale_1,
        tau_dof=cfg.model_cfg.sics_dof,
        tau_scale=cfg.model_cfg.sics_scale,
        read_error_model=error_model
    )

    return model


def get_input_paths(base_dir: Path) -> Tuple[List[Path], List[float]]:
    time_points = []
    read_paths = []

    input_specification_path = base_dir / "input_files.csv"
    try:
        with open(input_specification_path, "r") as f:
            input_specs = csv.reader(f, delimiter=',', quotechar='"')
            for item in input_specs:
                time_point_str, filename = item
                time_points.append(float(time_point_str))
                read_paths.append(base_dir / filename)
    except FileNotFoundError:
        raise FileNotFoundError("Missing required file `input_files.csv` in directory {}.".format(base_dir)) from None

    return read_paths, time_points


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

    read_paths, time_points = get_input_paths(Path(args.reads_dir))

    # ==== Load reads and validate.
    if len(read_paths) != len(time_points):
        raise ValueError("There must be exactly one set of reads for each time point specified.")

    if len(time_points) != len(set(time_points)):
        raise ValueError("Specified sample times must be distinct.")

    logger.info("Loading time-series read files.")
    reads = TimeSeriesReads.load(
        time_points=time_points,
        file_paths=read_paths
    )
    read_len = args.read_length

    # ============ Create model instance
    model = create_model(
        population=population,
        window_size=read_len,
        time_points=time_points,
        disable_quality=not cfg.model_cfg.use_quality_scores
    )

    """
    Perform inference using the chosen method. Available choices: 'em', 'bbvi'.
    1) 'em' runs Expectation-Maximization. Saves the learned abundances and plots them.
    2) 'bbvi' runs black-box VI and saves the learned posterior parametrization (as tensors).
    More methods to be potentially added for experimentation.
    """

    cache_tag = CacheTag(
        file_paths=read_paths,
        use_quality=cfg.model_cfg.use_quality_scores,
    )

    # ============ Prepare for algorithm output.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir():
        raise RuntimeError("Filesystem error: out_dir argument points to something other than a directory.")

    if args.true_abundance_path is not None:
        true_abundance_path = Path(args.true_abundance_path)
    else:
        true_abundance_path = None

    # ============ Run the specified algorithm.
    if args.method == 'em':
        logger.info("Solving using Expectation-Maximization.")
        perform_em(
            reads=reads,
            model=model,
            out_dir=out_dir,
            abnd_out_file=args.abundances_file,
            ground_truth_path=true_abundance_path,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            iters=args.iters,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag,
            plot_format=args.plot_format
        )
    elif args.method == 'bbvi':
        logger.info("Solving using Black-Box Variational Inference.")
        perform_bbvi(
            model=model,
            reads=reads,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            iters=args.iters,
            num_samples=args.num_samples,
            ground_truth_path=true_abundance_path,
            learning_rate=args.learning_rate,
            cache_tag=cache_tag,
            plot_format=args.plot_format,
            out_dir=out_dir,
            do_training_animation=True
        )
    # elif args.method == 'bbvi_reparametrization':
    #     logger.info("Solving using Black-Box Variational Inference.")
    #     plots_path = out_dir / "plot.{}".format(args.plot_format)
    #     samples_path = out_dir / "samples.pt"
    #     perform_bbvi_reparametrization(
    #         model=model,
    #         reads=reads,
    #         disable_quality=not cfg.model_cfg.use_quality_scores,
    #         iters=args.iters,
    #         learning_rate=args.learning_rate,
    #         cache_tag=cache_tag,
    #         plot_format=args.plot_format,
    #         plot_out_path=plots_path,
    #         samples_path=samples_path,
    #         ground_truth_path=true_abundance_path,
    #         num_posterior_samples=args.num_posterior_samples
    #     )
    # elif args.method == 'vsmc':
    #     logger.info("Solving using Variational Sequential Monte-Carlo.")
    #     plots_path = out_dir / "plot.{}".format(args.plot_format)
    #     samples_path = out_dir / "samples.pt"
    #     perform_vsmc(
    #         model=model,
    #         reads=reads,
    #         disable_quality=not cfg.model_cfg.use_quality_scores,
    #         iters=args.iters,
    #         num_samples=args.num_samples,
    #         ground_truth_path=true_abundance_path,
    #         plots_out_path=plots_path,
    #         samples_out_path=samples_path,
    #         learning_rate=args.learning_rate,
    #         cache_tag=cache_tag,
    #         plot_format=args.plot_format
    #     )
    # elif args.method == 'vi':
    #     logger.info("Solving using Variational Inference (Second-order mean-field solution).")
    #     plots_path = out_dir / "plot.{}".format(args.plot_format)
    #     samples_path = out_dir / "samples.pt"
    #     perform_vi(
    #         model=model,
    #         reads=reads,
    #         disable_quality=not cfg.model_cfg.use_quality_scores,
    #         iters=args.iters,
    #         num_samples=args.num_samples,
    #         ground_truth_path=true_abundance_path,
    #         plots_out_path=plots_path,
    #         samples_out_path=samples_path,
    #         cache_tag=cache_tag,
    #         plot_format=args.plot_format
    #     )
    else:
        raise ValueError("{} is not an implemented method.".format(args.method))


def plot_elbos(out_path: Path, elbos: List[float], plot_format: str):
    fig, ax = plt.subplots()
    ax.plot(
        np.arange(1, len(elbos)+1, 1),
        elbos
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO")
    plt.savefig(out_path, format=plot_format)


def plot_training_animation(out_path: Path, n_frames, lowers, uppers, medians, model):
    fig = plt.figure(figsize=(15, 10), dpi=100)
    ax = plt.axes(xlim=(model.times[0] - 0.5, model.times[-1] + 0.5), ylim=(0, 1))

    lines = [
        ax.plot([], [], lw=2)[0]
        for _ in range(model.num_strains())
    ]
    fills = [
        ax.fill_between([], [], [], facecolor=lines[i].get_color())
        for i in range(model.num_strains())
    ]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        for s_idx in range(model.num_strains()):
            lines[s_idx].set_data(model.times, medians[s_idx][i])
            fills[s_idx].remove()
            fills[s_idx] = ax.fill_between(
                model.times,
                lowers[s_idx][i],
                uppers[s_idx][i],
                alpha=0.2,
                color=lines[s_idx].get_color()
            )
        return lines + fills

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1, blit=True)
    anim.save(str(out_path), writer='imagemagick')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise
