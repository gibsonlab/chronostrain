from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import pandas as pd

from chronostrain.model import Population
from chronostrain.model.io import load_abundances
from chronostrain.logging import create_logger
logger = create_logger(__name__)


def plot_abundances_comparison(
            inferred_abnd_path: Path,
            real_abnd_path: Path,
            plots_out_path: Path,
            draw_legend: bool,
            num_reads_per_time: List[int] = None,
            title: str = None,
            ylim: List[float] = None,
            font_size: int = 12,
            thickness: int = 1,
            img_format: str = "pdf"):
    """
    Plot the specified abundances, along with the specified ground truth abundances (both given by CSV files).

    :param inferred_abnd_path:
    :param real_abnd_path:
    :param plots_out_path:
    :param draw_legend:
    :param num_reads_per_time:
    :param title:
    :param ylim:
    :param font_size:
    :param thickness:
    :param img_format:
    :return:
    """

    real_df = (pd.read_csv(real_abnd_path)
               .assign(Truth="Real")
               .melt(id_vars=['T', "Truth"],
                     var_name="Strain",
                     value_name="Abundance")
               .rename(columns={"T": "Time"}))

    inferred_df = (pd.read_csv(inferred_abnd_path)
                   .assign(Truth="Inferred")
                   .melt(id_vars=['T', "Truth"],
                         var_name="Strain",
                         value_name="Abundance")
                   .rename(columns={"T": "Time"}))

    result_df = pd.concat([real_df, inferred_df]).reset_index()
    plot_abundance_dataframe(
        data=result_df,
        plots_out_path=plots_out_path,
        draw_legend=draw_legend,
        num_reads_per_time=num_reads_per_time,
        title=title,
        ylim=ylim,
        font_size=font_size,
        thickness=[thickness, thickness],
        img_format=img_format
    )


def plot_abundances(
        abnd_path: Path,
        plots_out_path: Path,
        draw_legend: bool,
        num_reads_per_time: List[int] = None,
        title: str = None,
        ylim: List[float] = None,
        font_size: int = 12,
        thickness: int = 1,
        img_format: str = "pdf"):
    """
    Plot the specified abundances (given by a CSV file).

    :param abnd_path:
    :param plots_out_path:
    :param draw_legend:
    :param num_reads_per_time:
    :param title:
    :param ylim:
    :param font_size:
    :param thickness:
    :param img_format:
    :return:
    """
    inferred_df = (pd.read_csv(abnd_path)
                   .assign(Truth="Real")
                   .melt(id_vars=['T', "Truth"],
                         var_name="Strain",
                         value_name="Abundance")
                   .rename(columns={"T": "Time"})).reset_index()
    plot_abundance_dataframe(
        data=inferred_df,
        plots_out_path=plots_out_path,
        draw_legend=draw_legend,
        num_reads_per_time=num_reads_per_time,
        title=title,
        ylim=ylim,
        font_size=font_size,
        thickness=[thickness],
        img_format=img_format
    )


def plot_abundance_dataframe(
        data: pd.DataFrame,
        plots_out_path: Path,
        draw_legend: bool,
        img_format: str,
        num_reads_per_time: List[int] = None,
        title: str = None,
        ylim: List[float] = None,
        font_size: int = 12,
        thickness=(1, 1)):
    """
    Plot the abundance dataframe.
    Meant to be a helper for `plot_abundances` and `plot_abundances_comparison`.

    :param data:
    :param plots_out_path:
    :param draw_legend:
    :param img_format:
    :param num_reads_per_time:
    :param title:
    :param ylim:
    :param font_size:
    :param thickness:
    :return:
    """
    plt.rcParams.update({'font.size': font_size})
    ax = sns.lineplot(x="Time",
                      y="Abundance",
                      hue="Strain",
                      data=data,
                      style="Truth",
                      markers=True,
                      legend='full' if draw_legend else False,
                      size="Truth",
                      sizes=thickness)
    if ylim is None:
        ax.set_ylim([0.0, 1.0])
    else:
        ax.set_ylim(ylim)
    xlim = [data['Time'].min(), data['Time'].max()]
    xlim[0] = xlim[0] - (xlim[1] - xlim[0]) * 0.05
    xlim[1] = xlim[1] + (xlim[1] - xlim[0]) * 0.05
    ax.set_xlim(xlim)

    ax.set_xticks(data.Time.values)
    if title is not None:
        plt.title(title)
    if num_reads_per_time is not None:
        render_read_counts(data, num_reads_per_time, ax)

    plt.savefig(plots_out_path, bbox_inches='tight', format=img_format)


def plot_posterior_abundances(
        times: List[float],
        posterior_samples: np.ndarray,
        population: Population,
        plots_out_path: Path,
        draw_legend: bool,
        img_format: str,
        strain_trunc_level: float = 0.0,
        truth_path: Optional[Path] = None,
        title: str = None,
        cmap=sns.blend_palette(["firebrick", "palegreen"], 8),
        font_size: int = 12,
        thickness: int = 2,
        dpi: int = 100,
        width: int = 16,
        height: int = 10
):
    """
    :param times:
    :param posterior_samples: A (T x N x S) array of time-indexed samples of abundances.
    :param population:
    :param plots_out_path:
    :param truth_path:
    :param draw_legend:
    :param img_format:
    :param title:
    :param font_size:
    :param thickness:
    :param dpi:
    :return:
    """
    true_abundances = None
    if truth_path is not None:
        _, true_abundances, accessions = load_abundances(truth_path)
        truth_strain_id_to_idx = {
            acc: i
            for i, acc in enumerate(accessions)
        }
    else:
        truth_strain_id_to_idx = {}

    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    legend_elements = []
    plt.rcParams.update({'font.size': font_size})
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative abundance")

    ground_truth_colors = {}

    for truth_strain_id, truth_strain_idx in truth_strain_id_to_idx.items():
        true_trajectory = np.array([
            abundance_t[truth_strain_idx].item()
            for abundance_t in true_abundances
        ])
        color = render_single_abundance_trajectory(
            times=times,
            abundances=true_trajectory,
            ax=ax,
            linestyle='--',
            thickness=thickness * 2
        )
        ground_truth_colors[truth_strain_id] = color

    # but setting the number of colors explicitly allows it to use them all
    sns.set_palette(cmap, n_colors=population.num_strains())

    for s_idx, strain in enumerate(population.strains):
        # This is (T x N), for the particular strain.
        traj_samples = posterior_samples[:, :, s_idx]

        label = strain.id
        render_posterior_abundances(
            times=times,
            label=label,
            traj_samples=traj_samples,
            strain_trunc_level=strain_trunc_level,
            ax=ax,
            thickness=thickness,
            legend_elements=legend_elements,
            color=ground_truth_colors.get(strain.id, None)
        )

    if draw_legend:
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='lower center', handles=legend_elements)
        fig.tight_layout()
    ax.set_title(title)
    fig.savefig(plots_out_path, bbox_inches='tight', format=img_format, dpi=dpi)


def parse_quantiles(traj_samples: np.ndarray, quantiles: np.ndarray):
    # traj_samples: T x N
    return np.stack([
        np.quantile(traj_samples, q=q, axis=1)
        for q in quantiles
    ], axis=0)


def render_posterior_abundances(
        times: List[float],
        traj_samples: np.ndarray,
        label: str,
        ax,
        thickness: float,
        legend_elements: List,
        strain_trunc_level: float = 0.0,
        color: Optional = None,
        quantiles: Optional[np.ndarray] = None
):
    if quantiles is None:
        quantiles = np.array([0.025, 0.5, 0.975])
    if quantiles[0] > 0.5 or quantiles[-1] < 0.5:
        raise RuntimeError("Quantiles must lead with a value <= 0.5 and end with a value >= 0.5.")
    quantile_values = parse_quantiles(traj_samples, quantiles)  # size Q x T

    if np.sum(quantile_values[-1, :] > strain_trunc_level) == 0:
        return

    median = np.quantile(traj_samples, q=0.5, axis=1)

    # Plot the trajectory of medians.
    if color is None:
        line, = ax.plot(times, median, linestyle='--', marker='x', linewidth=thickness)
        color = line.get_color()
    else:
        ax.plot(times, median, linestyle='--', marker='x', linewidth=thickness, color=color)

    # # Plot subsampled trajectories.
    # for i in range(0, traj_samples.shape[1], 1000):
    #     ax.plot(times, traj_samples[:, i], linestyle='-', marker='o', linewidth=thickness, color=color)

    # Fill between the quantiles.
    for q_idx, (q, q_val) in enumerate(zip(quantiles, quantile_values)):
        if q < 0.5:
            alpha = 0.8 * (1 - (abs(q - 0.5) / 0.5))
            q_val_next = quantile_values[q_idx + 1]
            ax.fill_between(times, q_val, q_val_next, alpha=alpha, color=color, linewidth=0)
        if q > 0.5:
            alpha = 0.8 * (1 - (abs(q - 0.5) / 0.5))
            q_val_prev = quantile_values[q_idx - 1]
            ax.fill_between(times, q_val_prev, q_val, alpha=alpha, color=color, linewidth=0)

    # Populate the legend.
    legend_elements.append(
        Line2D([0], [0], color=color, lw=2, label=label)
    )
    return


def render_single_abundance_trajectory(
        times: List[float],
        abundances: np.ndarray,
        ax,
        thickness: float,
        linestyle: str = '-',
        color: Optional = None
):
    if color is None:
        line, = ax.plot(times, abundances, linestyle=linestyle, marker='o', linewidth=thickness)
        color = line.get_color()
    else:
        ax.plot(times, abundances, linestyle=linestyle, marker='o', color=color, linewidth=thickness)
    return color


def render_read_counts(dataframe: pd.DataFrame,
                       num_reads_per_time: List[int],
                       ax: plt.Axes):
    # Twin axes for including read count labels.
    ax2 = ax.twiny()
    sns.lineplot(x="Time", y="Abundance", hue="Strain", data=dataframe, style="Truth", markers=True, visible=False)
    ax2.get_legend().remove()

    # Move twinned axis ticks and label from top to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax2.spines["bottom"].set_position(("axes", -0.13))

    # Set tick position
    ax2.set_xticks(dataframe.Time.values)
    # Set tick labels
    ax2.set_xticklabels(num_reads_per_time)
    # Set axis label
    ax2.set_xlabel("# Reads")


def plot_elbo(elbo: np.ndarray,
              x_label: str,
              y_label: str,
              title: str,
              output_path: str,
              plot_format: str = "pdf"):
    """
    Plots the elbo values; can be used to check if the algs is working properly or not.
    :param elbo: The array of ELBO values.
    :param x_label: The x label to insert into the plot.
    :param y_label: The y label to insert into the plot.
    :param title: The title of the plot.
    :param output_path: The file path to save the plot to.
    :param plot_format: The file format to save the plot to. (example: "pdf", "png", "svg")
    """
    fig = plt.figure()
    axes = fig.ad_subplot(1, 1, 1)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)

    x = np.linspace(1, len(elbo), len(elbo))
    axes.plot(x, elbo)
    fig.savefig(output_path, format=plot_format)
