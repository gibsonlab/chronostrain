"""
  run_inference.py
  Run to perform inference on specified reads.
"""
import argparse
from pathlib import Path

from chronostrain.algs.subroutines.cache import ReadsComputationCache
from chronostrain import cfg, create_logger
import chronostrain.visualizations as viz
from chronostrain.algs.subroutines.alignments import CachedReadMultipleAlignments
from chronostrain.database import StrainDatabase
from chronostrain.model import Population, construct_fragment_space_uniform_length, FragmentSpace
from chronostrain.model.io import TimeSeriesReads

from helpers import *
logger = create_logger("chronostrain.run_inference")


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-r', '--reads_dir', required=True, type=str,
                        help='<Required> Directory containing read files. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')
    parser.add_argument('-m', '--method',
                        choices=['em', 'bbvi', 'bbvi_reparametrization'],
                        required=True,
                        help='<Required> A keyword specifying the inference method.')
    parser.add_argument('-l', '--read_length', required=False, type=int,
                        help='<Situationally Required> Specify the length of each read. Required if configuration'
                             'is set to use dense computation, which assumes uniform read lengths.')
    parser.add_argument('--input_file', required=False, type=str,
                        default='input_files.csv',
                        help='<Optional> The CSV input file specifier inside reads_dir.')

    # Output specification.
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The file path to save learned outputs to.')

    # Other Optional params
    parser.add_argument('-q', '--quality_format', required=False, type=str, default='fastq',
                        help='<Optional> The quality format. Should be one of the options implemented in Biopython '
                             '`Bio.SeqIO.QualityIO` module.')
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-truth', '--true_abundance_path', required=False, type=str,
                        help='<Optional> The CSV file path containing the ground truth relative abundances for each '
                             'strain by time point. For benchmarking.')
    parser.add_argument('--iters', required=False, type=int, default=10000,
                        help='<Optional> The number of iterations to run, if using EM or VI. Default: 10000')
    parser.add_argument('--num_samples', required=False, type=int, default=100,
                        help='<Optional> The number of samples to use for monte-carlo estimation '
                             '(for Variational solution).')
    parser.add_argument('--frag_chunk_size', required=False, type=int, default=100,
                        help='<Optional> The size of matrices to divide into chunks across fragments. (Default: 100)')
    parser.add_argument('-lr', '--learning_rate', required=False, type=float, default=1e-5,
                        help='<Optional> The learning rate to use for the optimizer, if using EM or VI. Default: 1e-5.')
    parser.add_argument('--abundances_file', required=False, default='abundances.out',
                        help='<Optional> (For EM only) Specify the filename for the learned abundances. '
                             'The file format depends on the method. '
                             'The file is saved to the output directory, specified by the -o option.')
    parser.add_argument('--num_posterior_samples', required=False, type=int, default=5000,
                        help='<Optional> If using a variational method, specify the number of '
                             'samples to generate as output.')
    parser.add_argument('--plot_elbo', action="store_true",
                        help='If flag is set, then outputs plots of the ELBO history (if using BBVI).')
    parser.add_argument('--draw_training_history', action="store_true",
                        help='If flag is set, then outputs an animation of the BBVI training history.')
    parser.add_argument('--save_fragment_probs', action="store_true",
                        help='If flag is set, then save posterior fragment probabilities for valid reads.')
    parser.add_argument('--plot_format', required=False, type=str, default="pdf")
    parser.add_argument('--print_debug_every', required=False, type=int, default=50)

    return parser.parse_args()


def load_fragments(reads: TimeSeriesReads, db: StrainDatabase) -> FragmentSpace:
    cache = ReadsComputationCache(reads)
    return cache.call(
        relative_filepath="inference_fragments.pkl",
        fn=aligned_exact_fragments,
        call_args=[reads, db]
    )


def aligned_exact_fragments(reads: TimeSeriesReads, db: StrainDatabase) -> FragmentSpace:
    logger.info("Constructing fragments from multiple alignments.")
    multiple_alignments = CachedReadMultipleAlignments(reads, db)
    fragment_space = FragmentSpace()
    for multi_align in multiple_alignments.get_alignments(num_cores=cfg.model_cfg.num_cores):
        logger.debug(f"Constructing fragments for marker `{multi_align.canonical_marker.name}`.")

        for frag_entry in multi_align.all_mapped_fragments():
            marker, read, subseq, insertions, deletions, start_clip, end_clip, revcomp = frag_entry

            fragment_space.add_seq(
                subseq,
                metadata=f"({read.id}->{marker.id})"
            )
    return fragment_space


def main():
    logger.info("Pipeline for inference started.")
    args = parse_args()
    initialize_seed(args.seed)

    # ==== Create database instance.
    db = cfg.database_cfg.get_database()

    # ==== Parse input reads.
    logger.info("Loading time-series read files.")
    reads = parse_reads(
        Path(args.reads_dir) / args.input_file,
        quality_format=args.quality_format
    )
    time_points = [time_slice.time_point for time_slice in reads]

    # ==== Load Population instance from database info
    population = Population(strains=db.all_strains(), extra_strain=cfg.model_cfg.extra_strain)
    if cfg.model_cfg.use_sparse:
        fragments = load_fragments(reads, db)
    else:
        fragments = construct_fragment_space_uniform_length(args.read_length, population)

    # ============ Create model instance
    model = create_model(
        population=population,
        fragments=fragments,
        time_points=time_points,
        disable_quality=not cfg.model_cfg.use_quality_scores,
        db=db
    )

    """
    Perform inference using the chosen method. Available choices: 'em', 'bbvi'.
    1) 'em' runs Expectation-Maximization. Saves the learned abundances and plots them.
    2) 'bbvi' runs black-box VI and saves the learned posterior parametrization (as tensors).
    More methods to be potentially added for experimentation.
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

    # ============ Run the specified algorithm.
    if args.method == 'em':
        logger.info("Solving using Expectation-Maximization.")
        perform_em(
            db=db,
            reads=reads,
            model=model,
            out_dir=out_dir,
            abnd_out_file=args.abundances_file,
            ground_truth_path=true_abundance_path,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            iters=args.iters,
            learning_rate=args.learning_rate,
            plot_format=args.plot_format
        )
    elif args.method == 'bbvi':
        logger.info("Solving using Black-Box Variational Inference.")
        solver, posterior, elbo_history, (uppers, lowers, medians) = perform_bbvi(
            db=db,
            model=model,
            reads=reads,
            iters=args.iters,
            learning_rate=args.learning_rate,
            num_samples=args.num_samples,
            correlation_type='time',
            save_elbo_history=args.plot_elbo,
            save_training_history=args.draw_training_history,
            frag_chunk_sz=args.frag_chunk_size,
            print_debug_every=args.print_debug_every
        )

        if args.plot_elbo:
            viz.plot_elbo_history(
                elbos=elbo_history,
                out_path=out_dir / "elbo.{}".format(args.plot_format),
                plot_format=args.plot_format
            )

        if args.draw_training_history:
            viz.plot_training_animation(
                model=model,
                out_path=out_dir / "training.gif",
                upper_quantiles=uppers,
                lower_quantiles=lowers,
                medians=medians
            )

        if args.save_fragment_probs:
            viz.save_frag_probabilities(
                reads=reads,
                solver=solver,
                out_path=out_dir / "reads_to_frags.csv"
            )

        # ==== Finally, plot the posterior.
        viz.plot_bbvi_posterior(
            model=model,
            posterior=posterior,
            plot_path=out_dir / "plot.{}".format(args.plot_format),
            samples_path=out_dir / "samples.pt",
            plot_format=args.plot_format,
            ground_truth_path=true_abundance_path,
            num_samples=args.num_posterior_samples,
            draw_legend=True
        )
    else:
        raise ValueError("{} is not an implemented method.".format(args.method))


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        exit(1)
