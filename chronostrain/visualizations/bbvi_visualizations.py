from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

from chronostrain import logger
from chronostrain.algs import AbstractPosterior
from chronostrain.model import Population
from chronostrain.model.generative import GenerativeModel
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

    logger.info("Saved ADVI ELBO history plot to {}.".format(out_path))


def plot_training_animation(
        model: GenerativeModel,
        out_path: Path,
        upper_quantiles: List[List[float]],
        lower_quantiles: List[List[float]],
        medians: List[List[float]],
        elbo_history: List[float],
        backend_writer: str = 'imagemagick'
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), dpi=100)

    # ========== First axis (samples)
    samples_ax = axes[0]
    samples_ax.set_xlim((model.times[0] - 0.5, model.times[-1] + 0.5))
    samples_ax.set_ylim((0, 1))

    lines = [
        samples_ax.plot([], [], lw=2)[0]
        for _ in range(model.num_strains())
    ]

    fills = [
        samples_ax.fill_between([], [], [], facecolor=lines[i].get_color())
        for i in range(model.num_strains())
    ]

    # =========== Second axis (Elbo)
    samples_ax = axes[1]
    samples_ax.plot(
        np.arange(1, len(elbo_history) + 1, 1),
        elbo_history
    )
    samples_ax.set_xlabel("Iteration")
    samples_ax.set_ylabel("ELBO")
    scat = plt.scatter([0], [0], c='red',  linewidths=1., edgecolors='black', s=100)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines + fills + [scat]

    def animate(i):
        for s_idx in range(model.num_strains()):
            lines[s_idx].set_data(model.times, medians[s_idx][i])
            fills[s_idx].remove()
            fills[s_idx] = samples_ax.fill_between(
                model.times,
                lower_quantiles[s_idx][i],
                upper_quantiles[s_idx][i],
                alpha=0.2,
                color=lines[s_idx].get_color()
            )
        # Set x and y data...
        scat.set_offsets([i+1, elbo_history[i]])
        return lines + fills + [scat]

    from matplotlib import animation
    n_frames = len(upper_quantiles[0])
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1, blit=True)
    anim.save(str(out_path), writer=backend_writer)

    logger.info("Saved ADVI training history to {}.".format(out_path))


def plot_vi_posterior(times: List[float],
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
    samples = posterior.sample(num_samples).detach().cpu()
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
