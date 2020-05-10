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
            title: str,
            plots_out_path: str,
            draw_legend: bool):

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
    if draw_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)

    plt.savefig(plots_out_path, bbox_inches='tight')


def plot_abundances(
        abnd_path: str,
        title: str,
        plots_out_path: str,
        draw_legend: bool):

    inferred_df = (pd.read_csv(abnd_path)
                   .melt(id_vars=['T'],
                         var_name="Strain",
                         value_name="Abundance")
                   .rename(columns={"T": "Time"}))

    sns.lineplot(x="Time", y="Abundance",
                 hue="Strain", data=inferred_df,
                 markers=True, legend=draw_legend)
    if draw_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)

    plt.savefig(plots_out_path, bbox_inches='tight')


def plot_posterior_abundances(
        times: List[int],
        posterior: AbstractVariationalPosterior,
        population: Population,
        title: str,
        plots_out_path: str,
        truth_path: str,
        draw_legend: bool,
        num_samples: int = 10000):

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

    # logger.debug(data.groupby(['Time', 'Strain'])[['Abundance']].std())

    if true_abundances is not None:
        true_abundances = true_abundances[0:len(times)]  # TODO debugging.
        data = pd.concat([pd.DataFrame(np.array(
            [
                (times[t], strain.name, abundance_t[truth_acc_dict[strain.name]].item(), 'Real')
                for s, strain in enumerate(population.strains)
                for t, abundance_t in enumerate(true_abundances)
            ],
            dtype=[('Time', int), ('Strain', '<U20'), ('Abundance', float), ('Truth', '<U10')]
        )), data])

    sns.lineplot(
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
    plt.title(title)
    plt.savefig(plots_out_path, bbox_inches='tight')
