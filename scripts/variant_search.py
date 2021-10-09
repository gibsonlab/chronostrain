import argparse
import csv
from pathlib import Path
from typing import Tuple, List, Iterable

import torch

from chronostrain import logger, cfg
from chronostrain.algs import StrainVariant
from chronostrain.model import Population, PhredErrorModel, GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.algs.variants import StrainVariantComputer


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads, and perform a meta-algorithm"
                                                 "for searching for a locally optimal variant space.")

    # Input specification.
    parser.add_argument('-r', '--reads_dir', required=True, type=str,
                        help='<Required> Directory containing read files. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')
    parser.add_argument('-l', '--read_length', required=True, type=int,
                        help='<Required> Length of each read')

    # Output specification.
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The file path to save learned outputs to.')

    # Other Optional params
    parser.add_argument('-q', '--quality_format', required=False, type=str, default='fastq',
                        help='<Optional> The quality format. Should be one of the options implemented in Biopython '
                             '`Bio.SeqIO.QualityIO` module.')
    parser.add_argument('--input_file', required=False, type=str,
                        default='input_files.csv',
                        help='<Optional> The CSV input file specifier inside reads_dir.')
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-truth', '--true_abundance_path', required=False, type=str,
                        help='<Optional> The CSV file path containing the ground truth relative abundances for each '
                             'strain by time point. For benchmarking.')
    parser.add_argument('--iters', required=False, type=int, default=3000,
                        help='<Optional> The number of iterations to run per BBVI instance.')
    parser.add_argument('--num_samples', required=False, type=int, default=100,
                        help='<Optional> The number of monte-carlo samples for BBVI.')
    parser.add_argument('-lr', '--learning_rate', required=False, type=float, default=1e-4,
                        help='<Optional> The learning rate to use for the optimizer, if using EM or VI. Default: 1e-4.')
    # parser.add_argument('--abundances_file', required=False, default='abundances.out',
    #                     help='<Optional> Specify the filename for the learned abundances. '
    #                          'The file format depends on the method. '
    #                          'The file is saved to the output directory, specified by the -o option.')
    # parser.add_argument('--num_posterior_samples', required=False, type=int, default=5000,
    #                     help='<Optional> If using a variational method, specify the number of '
    #                          'samples to generate as output.')
    # parser.add_argument('--plot_format', required=False, type=str, default="pdf")

    return parser.parse_args()


def get_input_paths(base_dir: Path, input_filename) -> Tuple[List[Iterable[Path]], List[float]]:
    time_points = []
    read_paths = []

    input_specification_path = base_dir / input_filename
    try:
        with open(input_specification_path, "r") as f:
            input_specs = csv.reader(f, delimiter=',', quotechar='"')
            for row in input_specs:
                time_point_str = row[0]
                filenames = [base_dir / f for f in row[1:]]

                time_points.append(float(time_point_str))
                read_paths.append(filenames)
    except FileNotFoundError:
        raise FileNotFoundError("Missing required file `input_files.csv` in directory {}.".format(base_dir)) from None

    return read_paths, time_points


def create_model(population: Population,
                 window_size: int,
                 time_points: List[float]):
    """
    Simple wrapper for creating a generative model.
    @param population: The bacteria population.
    @param window_size: Fragment read length to use.
    @param time_points: List of time points for which samples are taken from.
    @return A Generative model object.
    """
    mu = torch.zeros(population.num_strains(), device=cfg.torch_cfg.device)
    error_model = PhredErrorModel(read_len=window_size)

    model = GenerativeModel(
        bacteria_pop=population,
        read_length=window_size,
        times=time_points,
        mu=mu,
        tau_1_dof=cfg.model_cfg.sics_dof_1,
        tau_1_scale=cfg.model_cfg.sics_scale_1,
        tau_dof=cfg.model_cfg.sics_dof,
        tau_scale=cfg.model_cfg.sics_scale,
        read_error_model=error_model
    )

    return model


def main():
    args = parse_args()

    logger.info("Inference started on read inputs {}.".format(
        args.reads_dir
    ))
    torch.manual_seed(args.seed)

    # ==== Create database instance.
    db = cfg.database_cfg.get_database()

    # ==== Load Population instance from database info
    population = Population(strains=db.all_strains(), extra_strain=cfg.model_cfg.extra_strain)

    read_sources, time_points = get_input_paths(Path(args.reads_dir), args.input_file)

    # ==== Load reads and validate.
    if len(read_sources) != len(time_points):
        raise ValueError("There must be exactly one set of reads for each time point specified.")

    if len(time_points) != len(set(time_points)):
        raise ValueError("Specified sample times must be distinct.")

    logger.info("Loading time-series read files.")
    reads = TimeSeriesReads.load(
        time_points=time_points,
        source_entries=read_sources,
        quality_format=args.quality_format
    )
    read_len = args.read_length

    # ============ Create model instance
    model = create_model(
        population=population,
        window_size=read_len,
        time_points=time_points
    )

    """
    Perform inference and search for the optimal variant space.
    """
    # ============ Prepare for algorithm output.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir():
        raise RuntimeError("Filesystem error: out_dir argument points to something other than a directory.")

    if args.true_abundance_path is not None:
        true_abundance_path = Path(args.true_abundance_path)
    else:
        true_abundance_path = None

    # ============ Run the algorithm.
    # algorithm = VariantSearchAlgorithm(
    #     base_model=model,
    #     reads=reads,
    #     optim_iters=args.iters,
    #     optim_mc_samples=args.num_samples,
    #     optim_kwargs={
    #         'lr': args.learning_rate,
    #         'betas': (0.9, 0.999),
    #         'eps': 1e-7,
    #         'weight_decay': 0.
    #     }
    # )

    # variant_population, likelihood, vi_solution = algorithm.perform_search()
    # logger.info("Result: {}".format(variant_population))

    # =============== Clustering approach.
    computer = StrainVariantComputer(
        db=db,
        reads=reads,
        quality_threshold=20,
        eig_lower_bound=1e-3,
        variant_distance_upper_bound=1e-5
    )

    variants: List[StrainVariant] = list(computer.construct_variants())
    variants.sort(reverse=True, key=lambda v: v.quality_evidence)
    for variant in variants:
        print(repr(variant))
    print("# of strain variants = {}".format(len(variants)))


if __name__ == "__main__":
    main()
