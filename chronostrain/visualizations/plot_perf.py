import math
from typing import List, Tuple

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from chronostrain.model.io import load_abundances


def plot_performance_degradation(
        trials: List[Tuple[str, int, str]],
        true_abundance_path: str,
        out_path: str,
        title: str = None,
        draw_legend: bool = True,
        font_size: int = 18,
        thickness: int = 1,
        legend_labels: List[str] = None,
        img_format="pdf"
):
    """
    :param trials: The list of tuples (ID, num_reads, abundance_csv_path)
    :param true_abundance_path: A path to the ground truth.
    :param out_path: the file path to save the plot to.
    :param title: The title of the figure (default: None).
    :param legend_labels:
    :param draw_legend:
    :param thickness:
    :param font_size:
    :param img_format:
    """
    true_abundances = load_abundances(true_abundance_path)[1]
    ids = set()

    abundance_diffs = []
    for (trial_id, num_reads, path) in trials:
        _, abundances, _ = load_abundances(path)
        # diff = (abundances - true_abundances).norm().item()

        hellinger = (abundances.sqrt() - true_abundances.sqrt()).pow(2).sum(dim=1).sqrt().mean() / math.sqrt(2)
        # print((abundances.sqrt() - true_abundances.sqrt()).pow(2).sum(dim=1).sqrt() / math.sqrt(2))

        abundance_diffs.append((trial_id, num_reads, hellinger))
        ids.add(trial_id)
    df = pd.DataFrame(np.array(
        abundance_diffs,
        dtype=[('Label', '<U10'), ('# Reads on Markers', int), ('Average Hellinger error', float)]
    ))
    # print(df)

    plt.rcParams.update({'font.size': font_size})

    sns.lineplot(
        x='# Reads on Markers',
        y='Average Hellinger error',
        hue='Label',
        ci='sd',
        data=df,
        legend='full' if draw_legend else False,
        palette='cubehelix',
        size='Label',
        sizes=[thickness for _ in ids],
    )

    legend = plt.legend()
    if legend_labels is not None:
        for i, label in enumerate(legend_labels):
            legend.get_texts()[i].set_text(label)

    if title:
        plt.title(title)
    plt.savefig(out_path, bbox_inches='tight', format=img_format)
