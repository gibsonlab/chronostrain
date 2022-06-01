import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import softmax

from chronostrain.algs import ADVISolver, ADVISolverFullPosterior, EMSolver
from chronostrain.database import StrainDatabase
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads, save_abundances
from chronostrain.visualizations import plot_abundances, plot_abundances_comparison

from chronostrain import logger


def perform_bbvi(
        db: StrainDatabase,
        model: GenerativeModel,
        reads: TimeSeriesReads,
        num_epochs: int,
        lr_decay_factor: float,
        lr_patience: int,
        iters: int,
        learning_rate: float,
        num_samples: int,
        min_lr: float = 1e-4,
        read_batch_size: int = 5000,
        correlation_type: str = "strain",
        save_elbo_history: bool = False,
        save_training_history: bool = False
):
    # ==== Run the solver.
    if correlation_type == 'full':
        logger.warning("Encountered `full` correlation type argument; "
                       "learning this posterior may lead to unstable/unreliable results. "
                       "Consider directly invoking `perform_bbvi_full_correlation` instead.")
    solver = ADVISolver(
        model=model,
        data=reads,
        correlation_type=correlation_type,
        db=db,
        read_batch_size=read_batch_size
    )

    callbacks = []
    uppers = [[] for _ in range(model.num_strains())]
    lowers = [[] for _ in range(model.num_strains())]
    medians = [[] for _ in range(model.num_strains())]
    elbo_history = []

    if save_training_history:
        def anim_callback(x_samples, uppers_buf, lowers_buf, medians_buf):
            # Plot VI posterior.
            abund_samples = softmax(x_samples, dim=2).cpu().detach().numpy()
            for s_idx in range(model.num_strains()):
                traj_samples = abund_samples[:, :, s_idx]  # (T x N)
                upper_quantile = np.quantile(traj_samples, q=0.975, axis=1)
                lower_quantile = np.quantile(traj_samples, q=0.025, axis=1)
                median = np.quantile(traj_samples, q=0.5, axis=1)
                uppers_buf[s_idx].append(upper_quantile)
                lowers_buf[s_idx].append(lower_quantile)
                medians_buf[s_idx].append(median)

        callbacks.append(lambda epoch, x_samples, elbo: anim_callback(x_samples, uppers, lowers, medians))

    if save_elbo_history:
        def elbo_callback(elbo, elbo_buf):
            elbo_buf.append(elbo)
        callbacks.append(lambda epoch, x_samples, elbo: elbo_callback(elbo, elbo_history))

    start_time = time.time()
    solver.solve(
        optimizer_class=torch.optim.Adam,
        optimizer_args={'lr': learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.0},
        iters=iters,
        num_epochs=num_epochs,
        num_samples=num_samples,
        min_lr=min_lr,
        lr_decay_factor=lr_decay_factor,
        lr_patience=lr_patience,
        callbacks=callbacks
    )
    end_time = time.time()
    logger.debug("Finished inference in {} sec.".format(
        (end_time - start_time)
    ))

    posterior = solver.posterior
    return solver, posterior, elbo_history, (uppers, lowers, medians)


def perform_bbvi_full_correlation(
        db: StrainDatabase,
        model: GenerativeModel,
        reads: TimeSeriesReads,
        num_epochs: int,
        lr_decay_factor: float,
        lr_patience: int,
        iters: int,
        learning_rate: float,
        num_samples: int,
        num_importance_samples: int,
        importance_batch_size: int,
        temp_dir: Path,
        min_lr: float = 1e-4,
        read_batch_size: int = 5000,
        partial_correlation_type: str = "strain",
        save_elbo_history: bool = False,
        save_training_history: bool = False):
    solver = ADVISolverFullPosterior(
        model=model,
        data=reads,
        partial_correlation_type=partial_correlation_type,
        db=db,
        read_batch_size=read_batch_size
    )

    callbacks = []
    uppers = [[] for _ in range(model.num_strains())]
    lowers = [[] for _ in range(model.num_strains())]
    medians = [[] for _ in range(model.num_strains())]
    elbo_history = []

    if save_training_history:
        def anim_callback(x_samples, uppers_buf, lowers_buf, medians_buf):
            # Plot VI posterior.
            abund_samples = softmax(x_samples, dim=2).cpu().detach().numpy()
            for s_idx in range(model.num_strains()):
                traj_samples = abund_samples[:, :, s_idx]  # (T x N)
                upper_quantile = np.quantile(traj_samples, q=0.975, axis=1)
                lower_quantile = np.quantile(traj_samples, q=0.025, axis=1)
                median = np.quantile(traj_samples, q=0.5, axis=1)
                uppers_buf[s_idx].append(upper_quantile)
                lowers_buf[s_idx].append(lower_quantile)
                medians_buf[s_idx].append(median)

        callbacks.append(lambda epoch, x_samples, elbo: anim_callback(x_samples, uppers, lowers, medians))

    if save_elbo_history:
        def elbo_callback(elbo, elbo_buf):
            elbo_buf.append(elbo)

        callbacks.append(lambda epoch, x_samples, elbo: elbo_callback(elbo, elbo_history))

    start_time = time.time()
    solver.solve(
        optimizer_class=torch.optim.Adam,
        optimizer_args={'lr': learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.0},
        iters=iters,
        num_epochs=num_epochs,
        num_bbvi_samples=num_samples,
        num_importance_samples=num_importance_samples,
        batch_size=importance_batch_size,
        min_lr=min_lr,
        lr_decay_factor=lr_decay_factor,
        lr_patience=lr_patience,
        callbacks=callbacks,
        temp_dir=temp_dir
    )
    end_time = time.time()
    logger.debug("Finished inference in {} sec.".format(
        (end_time - start_time)
    ))

    posterior = solver.posterior
    return solver, posterior, elbo_history, (uppers, lowers, medians)


def perform_em(
        db: StrainDatabase,
        reads: TimeSeriesReads,
        model: GenerativeModel,
        out_dir: Path,
        abnd_out_file: str,
        ground_truth_path: Path,
        disable_quality: bool,
        iters: int,
        learning_rate: float,
        plot_format: str
):

    # ==== Run the solver.
    solver = EMSolver(model,
                      reads,
                      db,
                      lr=learning_rate)
    abundances, var_1, var = solver.solve(
        iters=iters,
        print_debug_every=1000,
        thresh=1e-5,
        gradient_clip=1e5
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
