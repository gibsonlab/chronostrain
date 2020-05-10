from typing import List

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from algs.vi import AbstractVariationalPosterior
from model.bacteria import Population
from util.io.model_io import load_abundances
from util.torch import multi_logit


def plot_abundances_comparison(
            inferred_abnd_path: str,
            real_abnd_path: str,
            plots_out_path: str,
            draw_legend: bool,
            num_reads_per_time: List[int] = None,
            title: str = None):

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

    ax = sns.lineplot(x="Time", y="Abundance", hue="Strain",
                      data=result_df, style="Truth", markers=True,
                      legend=draw_legend)
    ax.set_xticks(result_df.Time.values)
    if draw_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if title is not None:
        plt.title(title)
    if num_reads_per_time is not None:
        render_read_counts(result_df, num_reads_per_time, ax)

    plt.savefig(plots_out_path, bbox_inches='tight')


def plot_abundances(
        abnd_path: str,
        plots_out_path: str,
        draw_legend: bool,
        num_reads_per_time: List[int] = None,
        title: str = None):

    inferred_df = (pd.read_csv(abnd_path)
                   .melt(id_vars=['T'],
                         var_name="Strain",
                         value_name="Abundance")
                   .rename(columns={"T": "Time"}))

    ax = sns.lineplot(x="Time", y="Abundance",
                      hue="Strain", data=inferred_df,
                      markers=True, legend=draw_legend)
    if draw_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if title is not None:
        plt.title(title)
    if num_reads_per_time is not None:
        render_read_counts(inferred_df, num_reads_per_time, ax)

    plt.savefig(plots_out_path, bbox_inches='tight')


def plot_posterior_abundances(
        times: List[int],
        posterior: AbstractVariationalPosterior,
        population: Population,
        plots_out_path: str,
        truth_path: str,
        draw_legend: bool,
        num_samples: int = 10000,
        num_reads_per_time: List[int] = None,
        title: str = None):

    true_abundances = None
    truth_acc_dict = None
    if truth_path:
        _, true_abundances, accessions = load_abundances(truth_path, torch_device=torch.device("cpu"))
        truth_acc_dict = {acc: i for i, acc in enumerate(accessions)}

    abundance_samples = [multi_logit(x_t, dim=1) for x_t in posterior.sample(num_samples=num_samples)]
    data = pd.DataFrame(np.array(
        [
            (times[t], strain.name, abundance_t[i, s].item(), 'Learned')
            for i in range(num_samples)
            for s, strain in enumerate(population.strains)
            for t, abundance_t in enumerate(abundance_samples)
        ],
        dtype=[('Time', int), ('Strain', '<U20'), ('Abundance', float), ('Truth', '<U10')]
    ))

    if true_abundances is not None:
        true_abundances = true_abundances[0:len(times)]  # TODO remove when done debugging.
        data = pd.concat([pd.DataFrame(np.array(
            [
                (times[t], strain.name, abundance_t[truth_acc_dict[strain.name]].item(), 'Real')
                for s, strain in enumerate(population.strains)
                for t, abundance_t in enumerate(true_abundances)
            ],
            dtype=[('Time', int), ('Strain', '<U20'), ('Abundance', float), ('Truth', '<U10')]
        )), data])

    ax = sns.lineplot(
        x='Time',
        y='Abundance',
        hue='Strain',
        ci=95,
        data=data,
        style="Truth",
        markers=True,
        legend=draw_legend
    )

    if draw_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if title is not None:
        plt.title(title)
    if num_reads_per_time is not None:
        render_read_counts(data, num_reads_per_time, ax)

    plt.savefig(plots_out_path, bbox_inches='tight')


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
    ax2.spines["bottom"].set_position(("axes", -0.10))

    # Set tick position
    ax2.set_xticks(dataframe.Time.values)
    # Set tick labels
    ax2.set_xticklabels(num_reads_per_time)
    # Set axis label
    ax2.set_xlabel("# Reads")
