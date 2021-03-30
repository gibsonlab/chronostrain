import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import multivariate_normal as mult
from typing import List

def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)

def plot_abundances(
    true_abund: np.array,
    bbvi_abund:np.array,
    mean: np.array,
    std: np.array,
    times: List[float],
    x_lab: str,
    y_lab: str,
    title: str,
    save_name:str):

    """plots the abundances along with the 5th and 95th percentiles"""

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.set_xlabel(x_lab)
    axes.set_ylabel(y_lab)
    axes.set_title(title)

    n_bugs = true_abund.shape[1] #assuming the array is of shape:n_times x n_bugs
    u_quartile_vi = []
    l_quartile_vi = []
    print("n_bugs", n_bugs, mean)

    for i in range(mean.shape[0]):
        print("i", i)
        u_vi = []
        l_vi = []
        mean_vi = mean[i]
        sigma_vi = np.identity(n_bugs) * (std[i] ** 2)
        samples_vi = np.random.multivariate_normal(mean_vi, sigma_vi , size = 5000)
        abundance_vi = softmax(samples_vi)

        for j in range(n_bugs):
            u_vi.append(np.quantile(abundance_vi[:, j], 0.95))
            l_vi.append(np.quantile(abundance_vi[:, j], 0.05))
        u_quartile_vi.append(u_vi)
        l_quartile_vi.append(l_vi)

    colors = ["red", "blue", "green", "brown", "orange"]
    u_quartile_vi = np.asarray(u_quartile_vi).T
    l_quartile_vi = np.asarray(l_quartile_vi).T

    for i in range(0, n_bugs):
        real = true_abund[:, i]
        vi = bbvi_abund[:, i]
        axes.plot(times, real, linestyle = "-", marker = "o", color = colors[i],
              label = "Bug "+ str(i+1) +  " true")
        axes.plot(times, vi, linestyle = "--", marker = "v", color = colors[i],
              label = "Bug "+ str(i+1) +  " BBVI")

        axes.fill_between(times, l_quartile_vi[i], u_quartile_vi[i],
             alpha = 0.2, color = colors[i])
    box = axes.get_position()
    #axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig.savefig(save_name +".pdf", bbox_inches="tight", dpi = 300)
    plt.show()

def plot_elbo(elbo:np.array, x_lab: str, y_lab: str, title: str, savename: str):

    """plots the elbo values; can be used to check if the inference is
       working properly or not"""

    fig = plt.figure()
    axes = fig.ad_subplot(1, 1, 1)
    axes.set_xlabel(x_lab)
    axes.set_ylabel(y_lab)
    axes.set_title(title)

    x = np.linspace(1, len(elbo), len(elbo))
    axes.plot(x, elbo)

    fig.savefig(savename + ".pdf")
    plt.show()
