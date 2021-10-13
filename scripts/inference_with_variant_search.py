import argparse
from pathlib import Path
from typing import Tuple, List

from chronostrain import logger, cfg
from chronostrain.algs import StrainVariant, BBVISolver
from chronostrain.database import StrainDatabase
from chronostrain.model import Population, GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.algs.variants import StrainVariantComputer

import chronostrain.visualizations as viz

from scripts.helpers import *


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
    parser.add_argument('--seed_with_database', action='store_true',
                        help='If flag is turned on, initialize search with database-seeded strains. Otherwise,'
                             'the seed is determined by the variant with highest correlation.')
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
    parser.add_argument('--num_posterior_samples', required=False, type=int, default=5000,
                        help='<Optional> If using a variational method, specify the number of '
                             'samples to generate as output.')
    parser.add_argument('--plot_format', required=False, type=str, default="pdf")

    return parser.parse_args()


def search_best_variant_solution(
        db: StrainDatabase,
        reads: TimeSeriesReads,
        read_len: int,
        time_points: List[float],
        num_iters: int,
        learning_rate: float,
        num_samples: int,
        seed_with_database: bool
) -> Tuple[List[StrainVariant], GenerativeModel, BBVISolver, float]:
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
        print(variant)
    print("# of proposal strain variants = {}".format(len(variants)))

    # =============== Iteratively evaluate each variant.
    original_strains = db.all_strains()

    best_variants = []
    best_data_ll_estimate = float('-inf')
    best_result: Tuple[GenerativeModel, BBVISolver] = (None, None)

    if seed_with_database:
        gen = range(len(variants) + 1)
    else:
        gen = range(1, len(variants) + 1)

    for n_top_variants in gen:
        if seed_with_database:
            included_variants = variants[:n_top_variants]
            population = Population(strains=original_strains + included_variants)
        else:
            included_variants = variants[:n_top_variants]
            population = Population(strains=included_variants)

        # ============ Create model instance
        model = create_model(
            population=population,
            window_size=read_len,
            time_points=time_points,
            disable_quality=False
        )

        solver, posterior, _, _ = perform_bbvi(
            db=db,
            model=model,
            reads=reads,
            iters=num_iters,
            learning_rate=learning_rate,
            num_samples=num_samples,
            correlation_type="strain",
            save_elbo_history=False,
            save_training_history=False
        )

        x_latent_mean = solver.gaussian_posterior.mean()
        prior_ll = model.log_likelihood_x(x_latent_mean)
        data_ll = model.data_likelihood(x_latent_mean, solver.data_likelihoods.matrices)
        posterior_ll_est = solver.gaussian_posterior.log_likelihood(x_latent_mean)
        data_ll_estimate = (data_ll + prior_ll - posterior_ll_est).item()

        if data_ll_estimate <= best_data_ll_estimate:
            logger.debug("Data LL didn't improve ({:.3f} --> {:.3f}). Terminating search at variants [{}].".format(
                best_data_ll_estimate,
                data_ll_estimate,
                best_variants
            ))
            return best_variants, model, best_result[1], best_data_ll_estimate
        else:
            best_variants = included_variants
            best_data_ll_estimate = data_ll_estimate
            best_result = (model, solver)

    raise RuntimeError("Unexpected behavior: Could not iterate through variants.")


def main():
    args = parse_args()

    logger.info("Inference started on read inputs {}.".format(
        args.reads_dir
    ))
    initialize_seed(args.seed)

    # ==== Create database instance.
    db = cfg.database_cfg.get_database()

    # ==== Load Population instance from database info
    read_sources, read_depths, time_points = get_input_paths(Path(args.reads_dir), args.input_file)

    # ==== Load reads.
    logger.info("Loading time-series read files.")
    reads = TimeSeriesReads.load(
        time_points=time_points,
        read_depths=read_depths,
        source_entries=read_sources,
        quality_format=args.quality_format
    )
    read_len = args.read_length

    # ============ Prepare for algorithm output.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir():
        raise RuntimeError("Filesystem error: out_dir argument points to something other than a directory.")

    # =============== Run the variant search + inference.
    variants, model, solver, likelihood = search_best_variant_solution(
        db=db,
        reads=reads,
        read_len=read_len,
        time_points=time_points,
        num_iters=args.iters,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        seed_with_database=args.seed_with_database
    )

    logger.info("Final variants: {}".format(variants))
    logger.info("Final likelihood = {}".format(likelihood))

    # =============== output BBVI result.
    if args.save_fragment_probs:
        viz.save_frag_probabilities(
            reads=reads,
            solver=solver,
            out_path=out_dir / "reads_to_frags.csv"
        )

    if args.true_abundance_path is not None:
        true_abundance_path = Path(args.true_abundance_path)
    else:
        true_abundance_path = None

    # ==== Finally, plot the posterior.
    viz.plot_bbvi_posterior(
        model=model,
        posterior=solver.gaussian_posterior,
        plot_path=out_dir / "plot.{}".format(args.plot_format),
        samples_path=out_dir / "samples.pt",
        plot_format=args.plot_format,
        ground_truth_path=true_abundance_path,
        num_samples=args.num_posterior_samples
    )


if __name__ == "__main__":
    main()
