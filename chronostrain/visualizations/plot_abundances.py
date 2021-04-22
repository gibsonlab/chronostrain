from pathlib import Path
from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from chronostrain.model.bacteria import Population
from chronostrain.model.io import load_abundances
from scipy.special import softmax


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

    result_df = pd.concat([real_df, inferred_df])
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
                   .rename(columns={"T": "Time"}))
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
        truth_path: Path = None,
        title: str = None,
        font_size: int = 12,
        thickness: int = 1,
        dpi: int = 100):
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
    truth_acc_dict = None
    if truth_path is not None:
        _, true_abundances, accessions = load_abundances(truth_path)
        truth_acc_dict = {acc: i for i, acc in enumerate(accessions)}

    # Convert gaussians to rel abundances.
    abundance_samples = softmax(posterior_samples, axis=2)

    fig, ax = plt.subplots(1, 1)
    legend_elements = []
    plt.rcParams.update({'font.size': font_size})

    for s_idx, strain in enumerate(population.strains):
        # This is (T x N), for the particular strain.
        traj_samples = abundance_samples[:, :, s_idx]

        upper_quantile = np.quantile(traj_samples, q=0.975, axis=1)
        lower_quantile = np.quantile(traj_samples, q=0.025, axis=1)
        median = np.quantile(traj_samples, q=0.5, axis=1)

        # Plot the trajectory of medians.
        line, = ax.plot(times, median, linestyle='--', marker='x', linewidth=thickness)
        color = line.get_color()

        # Fill between the quantiles.
        ax.fill_between(times, lower_quantile, upper_quantile, alpha=0.2, color=color)

        # Plot true trajectory, if available.
        if true_abundances is not None:
            true_trajectory = np.array([
                abundance_t[truth_acc_dict[strain.id]].item()
                for abundance_t in true_abundances
            ])
            ax.plot(times, true_trajectory, linestyle='-', marker='o', color=color, linewidth=thickness)

        # Populate the legend.
        legend_elements.append(
            Line2D([0], [0], color=color, lw=2, label=strain.id)
        )

    if draw_legend:
        ax.legend(handles=legend_elements)
    ax.set_title(title)
    fig.savefig(plots_out_path, bbox_inches='tight', format=img_format, dpi=dpi)


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


def plot_elbo(elbo: np.array,
              x_label: str,
              y_label: str,
              title: str,
              output_path: str,
              plot_format: str = "pdf"):
    """
    Plots the elbo values; can be used to check if the inference is working properly or not.
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
