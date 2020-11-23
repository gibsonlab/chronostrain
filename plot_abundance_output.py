import argparse

from util.io.logger import logger
from visualizations.plot_abundances import plot_abundances_comparison, plot_abundances


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-a', '--abundance_path', required=True, type=str,
                        help='A specification of the abundance file.')
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='The file path to save the output plot to.')
    parser.add_argument('-n', '--num_reads', required=False, type=int, nargs='+',
                        help='Number of reads at each time point.')

    # Optional args.
    parser.add_argument('-g', '--ground_truth_path', required=False, type=str)
    parser.add_argument('--title', required=False, type=str,
                        help='The title for the plot.')
    parser.add_argument('--font_size', required=False, type=int, default=22)
    parser.add_argument('--thickness', required=False, type=int, default=1)
    parser.add_argument('--ylim', required=False, type=float, nargs='+')

    return parser.parse_args()


def main():
    args = parse_args()

    num_reads = args.num_reads

    if args.ground_truth_path is not None:
        # title += "\nSquare-Norm Abundances Difference: " + str(round(abundance_diff, 3))
        plot_abundances_comparison(
            inferred_abnd_path=args.abundance_path,
            real_abnd_path=args.ground_truth_path,
            title=args.title,
            plots_out_path=args.output_path,
            draw_legend=False,
            num_reads_per_time=num_reads,
            font_size=args.font_size,
            thickness=args.thickness,
            ylim=args.ylim
        )
    else:
        plot_abundances(
            abnd_path=args.abundance_path,
            title=args.title,
            plots_out_path=args.output_path,
            draw_legend=False,
            font_size=args.font_size,
            thickness=args.thickness,
            ylim=args.ylim
        )
    logger.info("Plots saved to {}".format(args.output_path))


if __name__ == "__main__":
    main()
