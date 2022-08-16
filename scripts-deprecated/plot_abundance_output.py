import argparse

from chronostrain import logger
from chronostrain.visualizations.plot_abundances import plot_abundances_comparison, plot_abundances


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
    parser.add_argument('-fmt', '--format', required=False, type=str, default="pdf")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.ground_truth_path is not None:
        plot_abundances_comparison(
            inferred_abnd_path=args.abundance_path,
            real_abnd_path=args.ground_truth_path,
            title=args.title,
            plots_out_path=args.output_path,
            draw_legend=False,
            num_reads_per_time=args.num_reads,
            font_size=args.font_size,
            thickness=args.thickness,
            ylim=args.ylim,
            img_format=args.format
        )
    else:
        plot_abundances(
            abnd_path=args.abundance_path,
            title=args.title,
            plots_out_path=args.output_path,
            draw_legend=False,
            font_size=args.font_size,
            thickness=args.thickness,
            ylim=args.ylim,
            img_format=args.format
        )
    logger.info("Plots saved to {}".format(args.output_path))


if __name__ == "__main__":
    main()
