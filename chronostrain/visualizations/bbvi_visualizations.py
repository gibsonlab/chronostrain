from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from chronostrain import logger
from chronostrain.algs import AbstractPosterior, BBVISolverV2
from chronostrain.model import Population
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util import filesystem

from .plot_abundances import plot_posterior_abundances


def plot_elbo_history(
        elbos: Union[List[float], np.ndarray],
        out_path: Path,
        plot_format: str):
    fig, ax = plt.subplots()
    ax.plot(
        np.arange(1, len(elbos) + 1, 1),
        elbos
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO")
    plt.savefig(out_path, format=plot_format)

    logger.info("Saved BBVI ELBO history plot to {}.".format(out_path))


def plot_training_animation(
        model: GenerativeModel,
        out_path: Path,
        upper_quantiles: List[List[float]],
        lower_quantiles: List[List[float]],
        medians: List[List[float]],
        backend_writer: str = 'imagemagick'
):
    fig = plt.figure(figsize=(15, 10), dpi=100)
    ax = plt.axes(xlim=(model.times[0] - 0.5, model.times[-1] + 0.5), ylim=(0, 1))

    lines = [
        ax.plot([], [], lw=2)[0]
        for _ in range(model.num_strains())
    ]

    # Populate the legend.
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='lower center', handles=[
        Line2D([0], [0], color=line.get_color(), lw=2, label=strain.id)
        for line, strain in zip(lines, model.bacteria_pop.strains)
    ])

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
                lower_quantiles[s_idx][i],
                upper_quantiles[s_idx][i],
                alpha=0.2,
                color=lines[s_idx].get_color()
            )
        return lines + fills

    from matplotlib import animation
    n_frames = len(upper_quantiles[0])
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1, blit=True)
    anim.save(str(out_path), writer=backend_writer)

    logger.info("Saved BBVI training history to {}.".format(out_path))


def plot_bbvi_posterior(times: List[float],
                        population: Population,
                        posterior: AbstractPosterior,
                        plot_path: Path,
                        samples_path: Path,
                        plot_format: str,
                        ground_truth_path: Optional[Path] = None,
                        draw_legend: bool = False,
                        num_samples: int = 10000,
                        width: int = 16,
                        height: int = 10,
                        title: str = "Posterior relative abundances"):
    logger.info("Generating plot of posterior.")

    # Generate and save posterior samples.
    samples = posterior.sample(num_samples)
    torch.save(samples, samples_path)
    logger.info("Posterior samples saved to {}. [{}]".format(
        samples_path,
        filesystem.convert_size(samples_path.stat().st_size)
    ))

    # Plotting.
    plot_posterior_abundances(
        times=times,
        posterior_samples=samples.cpu().numpy(),
        population=population,
        title=title,
        plots_out_path=plot_path,
        truth_path=ground_truth_path,
        draw_legend=draw_legend,
        img_format=plot_format,
        width=width,
        height=height
    )

    logger.info("Posterior abundance plot saved to {}.".format(plot_path))


def save_frag_probabilities(
        reads: TimeSeriesReads,
        solver: BBVISolverV2,
        out_path: Path
):
    df_entries = []
    for t_idx, reads_t in enumerate(reads):
        # Some reads got trimmed, take that into account by asking data_likelihoods about what the "true" index is.
        for r_idx, read in reads_t:
            for fragment, frag_prob in solver.fragment_posterior.top_fragments(t_idx, r_idx, top=5):
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

    import pandas as pd
    pd.DataFrame(df_entries).set_index(["time_idx", "read_idx"]).to_csv(str(out_path), mode="w")
    logger.info("Saved read-to-fragment likelihoods to {} [{}].".format(
        out_path,
        filesystem.convert_size(out_path.stat().st_size)
    ))
