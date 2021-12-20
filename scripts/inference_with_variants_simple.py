import argparse
from pathlib import Path

from Bio import SeqIO

from chronostrain.config import create_logger, cfg
from chronostrain.algs import GloppVariantSolver
from chronostrain.model import FragmentSpace, Population, StrainVariant
import chronostrain.visualizations as viz

from helpers import *
logger = create_logger("variant_search")


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads, and perform a meta-algorithm"
                                                 "for searching for a locally optimal variant space.")

    # I/O specification.
    parser.add_argument('-r', '--reads_dir', required=True, type=str,
                        help='<Required> Directory containing read files. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The file path to save learned outputs to.')

    # Other required params
    parser.add_argument('-n', '--num_strands', required=False, type=int,
                        help='<Optional> The number of strands to assemble for each marker.')

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
    parser.add_argument('--num_posterior_samples', required=False, type=int, default=5000,
                        help='<Optional> If using a variational method, specify the number of '
                             'samples to generate as output.')
    parser.add_argument('--draw_training_history', action="store_true",
                        help='If flag is set, then outputs an animation of the BBVI training history.')
    parser.add_argument('--plot_elbo', action="store_true",
                        help='If flag is set, then outputs plots of the ELBO history (if using BBVI).')
    parser.add_argument('--save_fragment_probs', action="store_true",
                        help='If flag is set, then save posterior fragment probabilities for valid reads.')
    parser.add_argument('--plot_format', required=False, type=str, default="pdf")
    parser.add_argument('--true_abundances', required=False, type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    if not cfg.model_cfg.use_sparse:
        raise NotImplementedError("Inference with variant construction not supported with `use_sparse=False`. "
                                  "Change this in the configuration file.")

    logger.info("Inference started on read inputs {}.".format(
        args.reads_dir
    ))
    initialize_seed(args.seed)

    # ==== Create database instance.
    db = cfg.database_cfg.get_database()

    # ==== Load reads.
    logger.info("Loading time-series read files.")

    reads = parse_reads(
        Path(args.reads_dir) / args.input_file,
        quality_format=args.quality_format
    )
    time_points = [time_slice.time_point for time_slice in reads]

    # ============ Prepare for algorithm output.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir():
        raise RuntimeError("Filesystem error: out_dir argument points to something other than a directory.")

    variants = list(GloppVariantSolver(
        db=db,
        reads=reads,
        time_points=time_points,
        bbvi_iters=args.iters,
        bbvi_lr=args.learning_rate,
        bbvi_num_samples=args.num_samples,
        quality_lower_bound=30,
        variant_count_lower_bound=5,
        seed_with_database=False,
        num_strands=args.num_strands,
        glasso_standardize=True
    ).propose_variants([]))

    # Construct the fragment space.
    logger.info("Compiling observed fragments.")
    fragments = FragmentSpace()
    for strain_idx, strain_variant in enumerate(variants):
        logger.info("Processing strain {} of {}".format(
            strain_idx + 1,
            len(variants)
        ))
        # Add all fragments implied by this new strain variant.
        for marker_variant in strain_variant.variant_markers:
            for t_idx, time_slice in enumerate(reads):
                for read in time_slice:
                    for subseq, revcomp, insertions, deletions, start_clip, end_clip in marker_variant.subseq_from_read(read):
                        fragments.add_seq(
                            subseq,
                            metadata=f"{read.id}->{marker_variant.id}"
                        )

    model = create_model(
        population=Population(variants),
        fragments=fragments,
        time_points=time_points,
        disable_quality=not cfg.model_cfg.use_quality_scores
    )

    # ==== For each strain, output its marker gene sequence.
    for idx, strain in enumerate(model.bacteria_pop.strains):
        if not isinstance(strain, StrainVariant):
            logger.info(f"Not outputting base strain `{strain.id}` to disk.")
            continue
        out_path = out_dir / f"{idx}_{strain.base_strain}_variant.fasta"
        write_strain_to_disk(strain, out_path)

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
        print_debug_every=100
    )

    if args.draw_training_history:
        viz.plot_training_animation(
            model=model,
            out_path=out_dir / "training.gif",
            upper_quantiles=uppers,
            lower_quantiles=lowers,
            medians=medians
        )

    if args.plot_elbo:
        viz.plot_elbo_history(
            elbos=elbo_history,
            out_path=out_dir / "elbo.{}".format(args.plot_format),
            plot_format=args.plot_format
        )

    # ==== Finally, plot the posterior.
    viz.plot_bbvi_posterior(
        model=model,
        posterior=posterior,
        plot_path=out_dir / "plot.{}".format(args.plot_format),
        samples_path=out_dir / "samples.pt",
        plot_format=args.plot_format,
        num_samples=args.num_posterior_samples,
        ground_truth_path=args.true_abundances,
        draw_legend=True,
        width=16,
        height=12
    )


def write_strain_to_disk(strain: StrainVariant, out_path: Path):
    records = []
    for marker in strain.markers:
        records.append(marker.to_seqrecord(description=""))

    with open(out_path, 'w') as out_file:
        SeqIO.write(records, out_file, "fasta")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
        exit(1)
