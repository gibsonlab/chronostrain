"""
  run_inference.py
  Run to perform inference on specified reads.
"""
import argparse
import csv
from pathlib import Path
from typing import List

import torch

from chronostrain import cfg, create_logger
import chronostrain.visualizations as viz
from chronostrain.model import Population

logger = create_logger("chronostrain.plot_bbvi_samples")


def parse_args():
    parser = argparse.ArgumentParser(description="Render plot of BBVI samples.")

    # Input specification.
    parser.add_argument('-r', '--reads_input', required=True, type=str,
                        help='<Required> Directory containing read files. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')
    parser.add_argument('-s', '--samples_path', required=True, type=str,
                        help='<Required> Path to BBVI posterior samples (pytorch tensor).')
    parser.add_argument('-o', '--out_path', required=True, type=str,
                        help='<Required> The target output path of the plot.')

    # Optional params/flags
    parser.add_argument('-t', '--title', required=False, type=str,
                        default='Timeseries posterior abundances',
                        help='<Optional> Specify to customize the title of the plot.')
    parser.add_argument('--draw_legend', action='store_true',
                        help='<Optional> If flag is set, renders the legend.')
    parser.add_argument('--plot_format', required=False, type=str,
                        default='PDF')
    parser.add_argument('-w', '--width', required=False, type=int,
                        default=8)
    parser.add_argument('-h', '--height', required=False, type=int,
                        default=6)
    parser.add_argument('--dpi', required=False, type=int,
                        default=30)
    parser.add_argument('--strain_trunc_level', required=False, type=float,
                        default=0.0,
                        help='<Optional> Any strain whose relative abundances do not exceed this lower bound across'
                             'all timepoints will not be plotted.')
    return parser.parse_args()


def parse_inputs(input_spec_path: Path) -> List[float]:
    time_points_to_reads = {}
    with open(input_spec_path, "r") as f:
        input_specs = csv.reader(f, delimiter=',', quotechar='"')
        for row in input_specs:
            time_point = float(row[0])
            num_reads = int(row[1])
            read_path = Path(row[2])

            if not read_path.exists():
                raise FileNotFoundError(
                    "The input specification `{}` pointed to `{}`, which does not exist.".format(
                        str(input_spec_path),
                        read_path
                    ))

            if time_point not in time_points_to_reads:
                time_points_to_reads[time_point] = []

            time_points_to_reads[time_point].append((num_reads, read_path))

    time_points = sorted(time_points_to_reads.keys(), reverse=False)
    return time_points


def main():
    args = parse_args()

    # ==== Create database instance.
    db = cfg.database_cfg.get_database()

    time_points = parse_inputs(Path(args.reads_input))
    population = Population(strains=db.all_strains(), extra_strain=cfg.model_cfg.extra_strain)

    samples = torch.load(args.samples_path)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ==== Finally, plot the posterior.
    viz.plot_posterior_abundances(
        times=time_points,
        posterior_samples=samples.cpu().numpy(),
        strain_trunc_level=args.strain_trunc_level,
        population=population,
        title=args.title,
        plots_out_path=out_path,
        truth_path=None,
        draw_legend=args.draw_legend,
        img_format=args.plot_format,
        width=args.width,
        height=args.height,
        dpi=args.dpi
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        exit(1)
