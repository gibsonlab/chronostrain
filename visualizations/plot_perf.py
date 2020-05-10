from typing import List

import numpy as np
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from util.io.model_io import load_abundances


def plot_performance_degradation(
        read_depths: List[int],
        abundance_replicate_paths: List[List[str]],
        true_abundance_path: str,
        out_path: str,
        title: str = None
):
    """
    :param read_depths: A list of read depths.
    :param abundance_replicate_paths: A nested list of paths to abundance CSVs.
    Indexed in order of: 1) read depth, 2) trial for that read depth.
    :param true_abundance_path: A path to the ground truth.
    :param out_path: the file path to save the plot to.
    :param title: The title of the figure (default: None).
    """
    true_abundances = load_abundances(true_abundance_path)[1]

    abundance_diffs = []
    for i, read_depth in enumerate(read_depths):
        for path in abundance_replicate_paths[i]:
            _, abundances, _ = load_abundances(path, torch_device=torch.device("cpu"))
            diff = (abundances - true_abundances).norm().item()
            abundance_diffs.append((read_depth, diff))
    df = pd.DataFrame(np.array(
        abundance_diffs,
        dtype=[('Read count', int), ('L2 error', float)]
    ))

    sns.lineplot(
        x='Read count',
        y='L2 error',
        ci='sd',
        data=df,
        legend=False
    )

    if title:
        plt.title(title)
    plt.savefig(out_path, bbox_inches='tight')
