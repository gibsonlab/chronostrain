"""
  run_inference.py
  Run to perform inference on specified reads.
"""
import argparse
from pathlib import Path
from typing import List
import torch

from chronostrain.algs.subroutines.cache import ReadsComputationCache
from chronostrain import cfg, create_logger
import chronostrain.visualizations as viz
from chronostrain.algs.subroutines.alignments import CachedReadMultipleAlignments, CachedReadPairwiseAlignments
from chronostrain.database import StrainDatabase
from chronostrain.model import Population, FragmentSpace, Strain
from chronostrain.model.io import TimeSeriesReads

from helpers import *
logger = create_logger("chronostrain.run_bbvi")


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-r', '--reads_input', required=True, type=str,
                        help='<Required> Path to the reads input CSV file. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')

    # Output specification.
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The file path to save learned outputs to.')

    # Optional BBVI params
    parser.add_argument('--iters', required=False, type=int, default=50,
                        help='<Optional> The number of iterations to run per epoch. (Default: 50)')
    parser.add_argument('--epochs', required=False, type=int, default=500,
                        help='<Optional> The number of epochs. (Default: 500)')
    parser.add_argument('--decay_lr', required=False, type=float, default=0.25,
                        help='<Optional> The multiplicative factor to apply to the learning rate based on '
                             'ReduceLROnPlateau criterion. (Default: 0.25)')
    parser.add_argument('--lr_patience', required=False, type=int, default=10,
                        help='<Optional> The `patience` parameter that specifies how many epochs to tolerate '
                             'no observed improvements before decaying lr. (Default: 10)')
    parser.add_argument('--min_lr', required=False, type=float, default=1e-4,
                        help='<Optional> Stop the algorithm when the LR is below this threshold. (Default: 1e-4)')
    parser.add_argument('-lr', '--learning_rate', required=False, type=float, default=0.01,
                        help='<Optional> The learning rate to use for the optimizer. (Default: 0.01.)')
    parser.add_argument('-n', '--num_samples', required=False, type=int, default=100,
                        help='<Optional> The number of samples to use for monte-carlo estimation of gradients. (Default: 100)')
    parser.add_argument('-b', '--read_batch_size', required=False, type=int, default=5000,
                        help='<Optional> The size of matrices to divide into batches across reads. (Default: 5000)')
    parser.add_argument('-c', '--correlation_mode', required=False, type=str, default='full',
                        help='<Optional> The correlation mode for the posterior. Options are "full", "strain", and "time".'
                             'For example, "strain" means that the abundance posteriors will be correlated over strains, and factorized across time.')

    # Optional input params
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-truth', '--true_abundance_path', required=False, type=str,
                        help='<Optional> The CSV file path containing the ground truth relative abundances for each '
                             'strain by time point. For benchmarking.')

    # Arguments for second-pass mode.
    parser.add_argument('--second_pass', action='store_true',
                        help='If flag is set, then initializes BBVI in "second-pass" mode. Automatically searches'
                             'for a chronostrain posterior approximation using the full DB, restricts to the set of '
                             'strains that are 99.9% confident to be greater than 1/<db size> and re-runs inference.')

    # Optional output params
    parser.add_argument('--num_posterior_samples', required=False, type=int, default=5000,
                        help='<Optional> If using a variational method, specify the number of '
                             'samples to generate as output.')
    parser.add_argument('--plot_elbo', action="store_true",
                        help='If flag is set, then outputs plots of the ELBO history (if using BBVI).')
    parser.add_argument('--draw_training_history', action="store_true",
                        help='If flag is set, then outputs an animation of the BBVI training history.')
    parser.add_argument('--plot_format', required=False, type=str, default="pdf")
    parser.add_argument('--single_ended', action='store_true')

    return parser.parse_args()


def load_fragments(reads: TimeSeriesReads, db: StrainDatabase) -> FragmentSpace:
    cache = ReadsComputationCache(reads)
    return cache.call(
        relative_filepath="inference_fragments.pkl",
        fn=aligned_exact_fragments,
        call_args=[reads, db]
    )


def aligned_exact_fragments(reads: TimeSeriesReads, db: StrainDatabase, mode: str = 'pairwise') -> FragmentSpace:
    logger.info("Constructing fragments from alignments.")
    fragment_space = FragmentSpace()

    if mode == 'pairwise':
        alignments = CachedReadPairwiseAlignments(reads, db)
        for _, aln in alignments.get_alignments():
            # First, add the likelihood for the fragment for the aligned base marker.
            fragment_space.add_seq(
                aln.marker_frag
            )

            if len(aln.marker_frag) < 15:
                raise Exception("UNEXPECTED ERROR! found frag of length smaller than 15")
    elif mode == 'multiple':
        multiple_alignments = CachedReadMultipleAlignments(reads, db)
        for multi_align in multiple_alignments.get_alignments(num_cores=cfg.model_cfg.num_cores):
            logger.debug(f"Constructing fragments for marker `{multi_align.canonical_marker.name}`.")

            for frag_entry in multi_align.all_mapped_fragments():
                marker, read, subseq, insertions, deletions, start_clip, end_clip, revcomp = frag_entry

                fragment_space.add_seq(
                    subseq,
                    metadata=f"({read.id}->{marker.id})"
                )
    else:
        raise ValueError(f"Unknown fragment extrapolation mode `{mode}`.")
    return fragment_space


def second_pass_strain_subset(db: StrainDatabase, pass1_samples_path: Path) -> List[Strain]:
    if not pass1_samples_path.exists():
        raise FileNotFoundError(
            f"Initial chronostrain run `{pass1_samples_path}` not found. "
            "Make sure that the first run has finished and that out_dir is pointing to the right directory."
        )

    abundance_samples = torch.softmax(torch.load(pass1_samples_path), dim=2)
    quantile_lower = torch.quantile(abundance_samples, q=0.001, dim=1)
    filtered_strain_idx = torch.nonzero(
        torch.sum(quantile_lower > 1 / db.num_strains(), dim=0),
        as_tuple=True
    )[0]

    all_strains = db.all_strains()
    return [all_strains[i] for i in filtered_strain_idx]


def main():
    logger.info("Pipeline for inference started.")
    args = parse_args()
    initialize_seed(args.seed)

    # ============ Create database instance.
    db = cfg.database_cfg.get_database()

    # ============ Prepare for algorithm output.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir():
        raise RuntimeError("Filesystem error: out_dir argument points to something other than a directory.")

    if args.true_abundance_path is not None:
        true_abundance_path = Path(args.true_abundance_path)
    else:
        true_abundance_path = None

    if args.second_pass:
        elbo_path = out_dir / "elbo.pass2.{}".format(args.plot_format)
        animation_path = out_dir / "training.pass2.gif"
        plot_path = out_dir / "plot.pass2.{}".format(args.plot_format)
        samples_path = out_dir / "samples.pass2.pt"
        samples_path_pass1 = out_dir / "samples.pass2.pt"
        strains_path = out_dir / "strains.pass2.txt"
        model_out_path = out_dir / "posterior.pass2.pt"

        strain_subset = second_pass_strain_subset(db, samples_path_pass1)
        population = Population(strains=strain_subset)
    else:
        elbo_path = out_dir / "elbo.{}".format(args.plot_format)
        animation_path = out_dir / "training.gif"
        plot_path = out_dir / "plot.{}".format(args.plot_format)
        samples_path = out_dir / "samples.pt"
        strains_path = out_dir / "strains.txt"
        model_out_path = out_dir / "posterior.pt"

        population = Population(strains=db.all_strains())

    # ============ Parse input reads.
    logger.info("Loading time-series read files.")
    reads = TimeSeriesReads.load_from_csv(
        Path(args.reads_input),
    )
    time_points = [time_slice.time_point for time_slice in reads]
    fragments = load_fragments(reads, db)

    # ============ Create model instance
    model = create_model(
        population=population,
        fragments=fragments,
        time_points=time_points,
        disable_quality=not cfg.model_cfg.use_quality_scores,
        db=db,
        pair_ended=not args.single_ended
    )

    """
    Perform inference using the chosen method. Available choices: 'em', 'bbvi'.
    1) 'em' runs Expectation-Maximization. Saves the learned abundances and plots them.
    2) 'bbvi' runs black-box VI and saves the learned posterior parametrization (as tensors).
    More methods to be potentially added for experimentation.
    """

    solver, posterior, elbo_history, (uppers, lowers, medians) = perform_bbvi(
        db=db,
        model=model,
        reads=reads,
        num_epochs=args.epochs,
        iters=args.iters,
        min_lr=args.min_lr,
        lr_decay_factor=args.decay_lr,
        lr_patience=args.lr_patience,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        correlation_type=args.correlation_mode,
        save_elbo_history=args.plot_elbo,
        save_training_history=args.draw_training_history,
        read_batch_size=args.read_batch_size,
    )

    if args.plot_elbo:
        viz.plot_elbo_history(
            elbos=elbo_history,
            out_path=elbo_path,
            plot_format=args.plot_format
        )

    if args.draw_training_history:
        viz.plot_training_animation(
            model=model,
            out_path=animation_path,
            upper_quantiles=uppers,
            lower_quantiles=lowers,
            medians=medians
        )

    # ==== Plot the posterior.
    viz.plot_bbvi_posterior(
        times=model.times,
        population=model.bacteria_pop,
        posterior=posterior,
        plot_path=plot_path,
        samples_path=samples_path,
        plot_format=args.plot_format,
        ground_truth_path=true_abundance_path,
        num_samples=args.num_posterior_samples,
        draw_legend=True
    )

    # ==== Output strain ordering.
    with open(strains_path, "w") as f:
        for strain in population.strains:
            print(strain.id, file=f)

    # ==== Save the posterior distribution.
    posterior.save(model_out_path)
    logger.debug(f"Saved model to `{model_out_path}`.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        exit(1)
