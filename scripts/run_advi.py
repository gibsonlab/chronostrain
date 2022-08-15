"""
  run_inference.py
  Run to perform inference on specified reads.
"""
import argparse
from pathlib import Path

from chronostrain.algs.subroutines.cache import ReadsComputationCache
import chronostrain.visualizations as viz
from chronostrain.algs.subroutines.alignments import CachedReadMultipleAlignments, CachedReadPairwiseAlignments
from chronostrain.database import StrainDatabase
from chronostrain.model import Population, FragmentSpace
from chronostrain.model.io import TimeSeriesReads

from helpers import *
from chronostrain.config import cfg
from chronostrain.logging import create_logger
logger = create_logger("chronostrain.run_advi")


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-r', '--reads_input', required=True, type=str,
                        help='<Required> Path to the reads input CSV file. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')

    # Output specification.
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The file path to save learned outputs to.')

    # Optional ADVI params
    parser.add_argument('--iters', required=False, type=int, default=50,
                        help='<Optional> The number of iterations to run per epoch. (Default: 50)')
    parser.add_argument('--epochs', required=False, type=int, default=1000,
                        help='<Optional> The number of epochs. (Default: 500)')
    parser.add_argument('--decay_lr', required=False, type=float, default=0.25,
                        help='<Optional> The multiplicative factor to apply to the learning rate based on '
                             'ReduceLROnPlateau criterion. (Default: 0.25)')
    parser.add_argument('--lr_patience', required=False, type=int, default=5,
                        help='<Optional> The `patience` parameter that specifies how many epochs to tolerate '
                             'no observed improvements before decaying lr. (Default: 5)')
    parser.add_argument('--min_lr', required=False, type=float, default=1e-5,
                        help='<Optional> Stop the algorithm when the LR is below this threshold. (Default: 1e-4)')
    parser.add_argument('-lr', '--learning_rate', required=False, type=float, default=0.01,
                        help='<Optional> The learning rate to use for the optimizer. (Default: 0.01.)')
    parser.add_argument('-n', '--num_samples', required=False, type=int, default=200,
                        help='<Optional> The number of samples to use for monte-carlo estimation of gradients. (Default: 100)')
    parser.add_argument('-b', '--read_batch_size', required=False, type=int, default=2500,
                        help='<Optional> The size of matrices to divide into batches across reads. (Default: 5000)')
    parser.add_argument('-c', '--correlation_mode', required=False, type=str, default='full',
                        help='<Optional> The correlation mode for the posterior. Options are "full", "strain", '
                             'and "time". For example, "strain" means that the abundance posteriors will be correlated '
                             'over strains, and factorized across time.')

    parser.add_argument('--full_corr_num_importance_samples', required=False, type=int, default=50000,
                        help='<Optional> The number of importance samples to use to '
                             'estimate the full posterior covariance/mean.'
                             'Only used if `--correlation_mode` is set to `full`.')
    parser.add_argument('--full_corr_importance_batch_size', required=False, type=int, default=1000,
                        help='<Optional> The number of importance samples to allocate into each batch. '
                             'Only used if `--correlation_mode` is set to `full`.')

    # Optional input params
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-truth', '--true_abundance_path', required=False, type=str,
                        help='<Optional> The CSV file path containing the ground truth relative abundances for each '
                             'strain by time point. For benchmarking.')

    # Optional output params
    parser.add_argument('--num_posterior_samples', required=False, type=int, default=5000,
                        help='<Optional> If using a variational method, specify the number of '
                             'samples to generate as output.')
    parser.add_argument('--plot_elbo', action="store_true",
                        help='If flag is set, then outputs plots of the ELBO history (if using ADVI).')
    parser.add_argument('--draw_training_history', action="store_true",
                        help='If flag is set, then outputs an animation of the ADVI training history.')
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
    fragments = load_fragments(reads, db)

    # ============ Create model instance
    solver, posterior, elbo_history, (uppers, lowers, medians) = perform_advi(
        db=db,
        population=population,
        fragments=fragments,
        reads=reads,
        paired_end=not args.single_ended,
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
            model=solver.model,
            out_path=animation_path,
            upper_quantiles=uppers,
            lower_quantiles=lowers,
            medians=medians
        )

    # ==== Plot the posterior.
    viz.plot_vi_posterior(
        times=solver.model.times,
        population=population,
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
