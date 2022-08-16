import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
from chronostrain.config import create_logger, cfg
from chronostrain.model import Strain

logger = create_logger("sample_reads")


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate reads from specified variant genomes, using ART.")

    # ============== Required params
    parser.add_argument('-o', '--output_dir', required=True, type=str,
                        help='<Required> The path to which the output abundances will be written.')
    parser.add_argument('-t', '--time_points', required=True, type=str,
                        help='<Required> A comma-separated list of time-point float values.')
    parser.add_argument('-n', '--num_strains', required=True, type=int,
                        help='<Required> The number of strains included in the sample.')

    # ============ Optional params
    parser.add_argument('-s', '--seed', dest='seed', required=False, type=int, default=random.randint(0, 100),
                        help='<Optional> The random seed to use for the samplers. Each timepoint will use a unique '
                             'seed, starting with the specified value and incremented by one at a time.')

    return parser.parse_args()


def generate_abundance_profile(time_points: List[float], strains: List[Strain]):
    time_points.append()


def main():
    args = parse_args()
    db = cfg.database_cfg.get_database()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    time_points = [float(t) for t in args.time_points.split(',')]



if __name__ == "__main__":
    main()
