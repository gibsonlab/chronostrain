import argparse
from pathlib import Path
import csv
from typing import List, Tuple

from chronostrain import logger
from chronostrain.model.io import load_abundances

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-t', '--trial_specification', required=True,
                        help='The path to the trial specification CSV file.')
    parser.add_argument('-g', '--ground_truth_path', required=True, type=str)
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='The file path to save the output plot to.')
    parser.add_argument('--title', required=False, type=str, default=None,
                        help='The title for the plot.')

    parser.add_argument('--font_size', required=False, type=int, default=22)
    parser.add_argument('--thickness', required=False, type=int, default=1)
    parser.add_argument('--draw_legend', action="store_true")
    parser.add_argument('--legend_labels', required=False, type=str, nargs='+')
    parser.add_argument('-fmt', '--format', required=False, type=str, default="pdf")
    parser.add_argument('--width', required=False, type=int, default=15)
    parser.add_argument('--height', required=False, type=int, default=10)

    return parser.parse_args()


def parse_trials(trials_filepath: Path) -> List[Tuple[str, int, Path]]:
    trials = []
    with open(trials_filepath, "r") as f:
        reader = csv.reader(f, delimiter=',', quotechar="\"")
        for row in reader:
            trial_id = row[0].strip()
            q_shift = int(row[1].strip())
            file_path = Path(row[2].strip())
            trials.append((trial_id, q_shift, file_path))
    return trials


def load_abundance_samples(filepath: Path):
    x = torch.load(filepath)
    return torch.softmax(x, dim=2)


def plot_performance_comparison(
        trials: List[Tuple[str, int, Path]],
        true_abundance_path: Path,
        out_path: Path,
        title: str = None,
        draw_legend: bool = True,
        font_size: int = 18,
        thickness: int = 1,
        width: int = 15,
        height: int = 10,
        legend_labels: List[str] = None,
        img_format="pdf"
):
    true_abundances = load_abundances(true_abundance_path)[1]  # (T x S)
    ids = set()

    abundance_diffs = []
    for (trial_id, q_shift, path) in trials:
        if path.suffix == ".csv":
            try:
                _, abundances, _ = load_abundances(path)
            except FileNotFoundError as e:
                logger.warn("File {} not found. Skipping.".format(path))
                continue
            hellinger = (abundances.sqrt() - true_abundances.sqrt()).pow(2).sum(dim=1).sqrt().mean() / np.sqrt(2)
            abundance_diffs.append((trial_id, q_shift, hellinger))
        elif path.suffix == ".pt":
            try:
                abundances = load_abundance_samples(path)  # (T x N x S)
            except FileNotFoundError as e:
                logger.warn("File {} not found. Skipping.".format(path))
                continue
            hellinger_errors = torch.pow(
                abundances.sqrt() - true_abundances.unsqueeze(1).sqrt(),
                2
            ).sum(dim=2).pow(0.5).mean(dim=0) / np.sqrt(2)  # length-N
            abundance_diffs.append((trial_id, q_shift, hellinger_errors.mean().item()))
        else:
            raise RuntimeError("File extension `{}` not recognized. (input filepath: {})".format(
                path.suffix,
                path
            ))

        ids.add(trial_id)
    df = pd.DataFrame(np.array(
        abundance_diffs,
        dtype=[('Label', '<U10'), ('Quality Shift', int), ('Hellinger Error', float)]
    ))

    plt.rcParams.update({'font.size': font_size})

    fig, ax = plt.subplot(figsize=(width, height))
    sns.boxplot(
        x='Quality Shift',
        y='Hellinger Error',
        hue='Label',
        data=df,
        palette='cubehelix',
        ax=ax
    )

    legend = ax.legend()
    if legend_labels is not None:
        for i, label in enumerate(legend_labels):
            legend.get_texts()[i].set_text(label)

    if title:
        ax.title(title)
    fig.savefig(out_path, bbox_inches='tight', format=img_format)


def main():
    args = parse_args()
    trials = parse_trials(Path(args.trial_specification))

    plot_performance_comparison(
        trials=trials,
        true_abundance_path=Path(args.ground_truth_path),
        out_path=Path(args.output_path),
        title=args.title,
        font_size=args.font_size,
        thickness=args.thickness,
        draw_legend=args.draw_legend,
        legend_labels=args.legend_labels,
        img_format=args.format
    )
    logger.info("Output the performance plot to {}".format(args.output_path))


if __name__ == "__main__":
    main()
