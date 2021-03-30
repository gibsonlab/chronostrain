import argparse
import csv
from typing import List, Tuple

from chronostrain import logger
from chronostrain.visualizations.plot_perf import plot_performance_degradation


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

    return parser.parse_args()


def parse_trials(trials_filepath: str) -> List[Tuple[str, int, str]]:
    trials = []
    with open(trials_filepath, "r") as f:
        reader = csv.reader(f, delimiter=',', quotechar="\"")
        for row in reader:
            trial_id = row[0].strip()
            num_reads = int(row[1].strip())
            file_path = row[2].strip()
            trials.append((trial_id, num_reads, file_path))
    return trials


def main():
    args = parse_args()
    trials = parse_trials(args.trial_specification)

    plot_performance_degradation(
        trials=trials,
        true_abundance_path=args.ground_truth_path,
        out_path=args.output_path,
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
