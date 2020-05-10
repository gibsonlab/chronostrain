import argparse
from util.io.logger import logger
from visualizations.plot_perf import plot_performance_degradation


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-t', '--trial', required=True, action='append', nargs=2,
                        help='A specification of a single trial: (num_reads, file_path). '
                             'Repeat to append.')
    parser.add_argument('-g', '--ground_truth_path', required=True, type='str')
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='The file path to save the output plot to.')
    parser.add_argument('--title', required=False, type=str, default=None,
                        help='The title for the plot.')
    return parser.parse_args()


def get_dir_structure(trial_list):
    read_depths = [int(trial[0]) for trial in trial_list]
    read_depths.sort()

    abundance_paths = [[] for _ in read_depths]
    depths_dict = {depth: i for i, depth in enumerate(read_depths)}

    for trial in trial_list:
        abundance_paths[
            depths_dict[int(trial[0])]
        ].append(
            trial[1]
        )

    return read_depths, abundance_paths


def main():
    logger.info("Pipeline for inference started.")
    args = parse_args()

    read_depths, abundance_paths = get_dir_structure(args.trial)
    plot_performance_degradation(
        read_depths=read_depths,
        abundance_replicate_paths=abundance_paths,
        true_abundance_path=args.ground_truth_path,
        out_path=args.output_path,
        title=args.title
    )
