from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from chronostrain.algs.vi import AbstractVariationalPosterior
from chronostrain.model.bacteria import Population
from chronostrain.model.io import load_abundances
from torch.nn.functional import softmax


def plot_abundances_comparison(
            inferred_abnd_path: str,
            real_abnd_path: str,
            plots_out_path: str,
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
        abnd_path: str,
        plots_out_path: str,
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
        plots_out_path: str,
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
        posterior: AbstractVariationalPosterior,
        population: Population,
        plots_out_path: str,
        truth_path: str,
        draw_legend: bool,
        img_format: str,
        num_samples: int = 500,
        num_reads_per_time: List[int] = None,
        title: str = None,
        font_size: int = 12,
        thickness: int = 1):

    true_abundances = None
    truth_acc_dict = None
    if truth_path:
        _, true_abundances, accessions = load_abundances(truth_path)
        truth_acc_dict = {acc: i for i, acc in enumerate(accessions)}

    abundance_samples = [
        softmax(x_t, dim=1)
        for x_t in posterior.sample(num_samples=num_samples)
    ]
    data = pd.DataFrame(np.array(
        [
            (times[t], strain.name, abundance_t[i, s].item(), 'Learned')
            for i in range(num_samples)
            for s, strain in enumerate(population.strains)
            for t, abundance_t in enumerate(abundance_samples)
        ],
        dtype=[('Time', float), ('Strain', '<U20'), ('Abundance', float), ('Truth', '<U10')]
    ))
    sizes = [thickness]

    if true_abundances is not None:
        true_abundances = true_abundances[0:len(times)]  # TODO remove when done debugging.
        data = pd.concat([pd.DataFrame(np.array(
            [
                (times[t], strain.name, abundance_t[truth_acc_dict[strain.name]].item(), 'Real')
                for s, strain in enumerate(population.strains)
                for t, abundance_t in enumerate(true_abundances)
            ],
            dtype=[('Time', float), ('Strain', '<U20'), ('Abundance', float), ('Truth', '<U10')]
        )), data])
        sizes = sizes + [thickness]

    data.to_csv("data/output/trivial_test/samples.csv")
    print("Generated {} samples for plotting.".format(num_samples))
    print(data)

    plt.rcParams.update({'font.size': font_size})
    ax = sns.lineplot(
        x='Time',
        y='Abundance',
        hue='Strain',
        ci=99,
        data=data,
        style="Truth",
        markers=True,
        legend='full' if draw_legend else False,
        size="Truth",
        sizes=sizes
    )
    if title is not None:
        plt.title(title)
    if num_reads_per_time is not None:
        render_read_counts(data, num_reads_per_time, ax)

    plt.savefig(plots_out_path, bbox_inches='tight', format=img_format)


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
