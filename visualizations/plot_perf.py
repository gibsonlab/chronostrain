from typing import List, Tuple

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from util.io.model_io import load_abundances


def plot_performance_degradation(
        trials: List[Tuple[str, int, str]],
        true_abundance_path: str,
        out_path: str,
        title: str = None,
        draw_legend: bool = True,
        font_size: int = 18,
        thickness: int = 1
):
    """
    :param trials: The list of tuples (ID, num_reads, abundance_csv_path)
    :param true_abundance_path: A path to the ground truth.
    :param out_path: the file path to save the plot to.
    :param title: The title of the figure (default: None).
    :param thickness:
    :param font_size:
    """
    true_abundances = load_abundances(true_abundance_path)[1]
    ids = set()

    abundance_diffs = []
    for (id, num_reads, path) in trials:
        _, abundances, _ = load_abundances(path)
        diff = (abundances - true_abundances).norm().item()
        abundance_diffs.append((id, num_reads, diff))
        ids.add(id)
    df = pd.DataFrame(np.array(
        abundance_diffs,
        dtype=[('Label', '<U10'), ('Read count', int), ('L2 error', float)]
    ))
    print(df)

    plt.rcParams.update({'font.size': font_size})

    ax = sns.lineplot(
        x='Read count',
        y='L2 error',
        hue='Label',
        ci='sd',
        data=df,
        legend='full' if draw_legend else False,
        size='Label',
        sizes=[thickness for _ in ids],
    )

    if title:
        plt.title(title)
    plt.savefig(out_path, bbox_inches='tight')
