from typing import List

import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from model.bacteria import Population
from util.io.model_io import load_abundances
from util.torch import multi_logit


def plot_abundances_comparison(
            inferred_abnd_path: str,
            real_abnd_path: str,
            title: str,
            plots_out_path: str):

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

    sns.lineplot(x="Time", y="Abundance", hue="Strain", data=result_df, style="Truth", markers=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)

    plt.savefig(plots_out_path, bbox_inches='tight')


def plot_abundances(
        abnd_path: str,
        title: str,
        plots_out_path: str):

    inferred_df = (pd.read_csv(abnd_path)
                   .melt(id_vars=['T'],
                         var_name="Strain",
                         value_name="Abundance")
                   .rename(columns={"T": "Time"}))

    sns.lineplot(x="Time", y="Abundance", hue="Strain", data=inferred_df, markers=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)

    plt.savefig(plots_out_path, bbox_inches='tight')


def plot_posterior_abundances(
        times: List[int],
        population: Population,
        means: List[torch.Tensor],
        variances: List[torch.Tensor],
        title: str,
        plots_out_path: str,
        truth_path: str):

    true_abundances = None
    truth_acc_dict = None
    if truth_path:
        _, true_abundances, accessions = load_abundances(truth_path)
        truth_acc_dict = {acc: i for i, acc in enumerate(accessions)}

    mean_softmax = [multi_logit(means[t], dim=0) for t in range(len(times))]
    stdevs = [variances[t].sqrt() for t in range(len(times))]

    for s, strain in enumerate(population.strains):
        val = [mean_softmax[t][s].item() for t in range(len(times))]
        # delta = -torch.ones(size=means[0].size(), device=means[0].device)
        # delta[s] = 1
        # upper = [
        #     multi_logit(means[t] + 2 * (stdevs[t] * delta), dim=0)[s].item()
        #     for t in range(len(times))
        # ]
        # lower = [
        #     multi_logit(means[t] - 2 * (stdevs[t] * delta), dim=0)[s].item()
        #     for t in range(len(times))
        # ]

        p = plt.plot(times, val, linestyle='solid', label=strain.name)
        color = p[-1].get_color()
        # plt.fill_between(times, lower, upper, facecolor=color, alpha=0.2)

        if true_abundances is not None:
            truth = [true_abundances[t, truth_acc_dict[strain.name]].item() for t in range(len(times))]
            plt.plot(times, truth, linestyle='dotted', color=color, label=strain.name + '(True)')

        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Abundance')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.savefig(plots_out_path, bbox_inches='tight')
