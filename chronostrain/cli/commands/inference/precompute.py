import click
from pathlib import Path
from ..base import option


@click.command()
@option(
    '--reads', '-r', 'reads_input',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the reads input CSV file."
)
@option(
    '--strain-subset', '-s', 'strain_subset_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=False, readable=True),
    required=False, default=None,
    help="A text file specifying a subset of database strain IDs to perform filtering with; "
         "a TSV file containing one ID per line, optionally with a second column for metadata.",
)
def main(
        reads_input: Path,
        strain_subset_path: Path,
):
    """
    Perform alignments and fragment counting to pre-calculate necessary model parameters prior to inference.
    This command should be run if it is necessary to perform these operations (which uses bowtie2,
    bwa-mem and/or bwa fastmap) separately prior to running ADVI inference.
    All results are stored into the cache directory.

    For instance, if one wants to run the model inference on a GPU compute cluster, then it might be beneficial to
    pre-compute the cpu-intensive tasks beforehand, which is what this command does.
    """
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli.precompute")

    logger.info("Pipeline for pre-computation of inference models started.")
    from chronostrain.config import cfg
    from chronostrain.model import TimeSeriesReads, StrainCollection
    from chronostrain.inference.likelihoods import ReadFragmentMappings, ReadStrainCollectionCache, FragmentFrequencyComputer
    from chronostrain.util.math import negbin_fit_frags
    import numpy as cnp
    from .helpers import create_error_model

    logger.info("Loading time-series read files from {}".format(reads_input))
    timeseries_reads = TimeSeriesReads.load_from_file(reads_input)
    if timeseries_reads.total_number_reads() == 0:
        logger.info("Input has zero reads. Terminating.")
        return 0

    db = cfg.database_cfg.get_database()

    if strain_subset_path is not None:
        with open(strain_subset_path, "rt") as f:
            strain_collection = StrainCollection(
                [db.get_strain(line.strip().split('\t')[0]) for line in f if not line.startswith("#")],
                db.signature
            )
        logger.info("Loaded list of {} strains.".format(len(strain_collection)))
    else:
        strain_collection = StrainCollection(db.all_strains(), db.signature)
        logger.info("Using complete collection of {} strains from database.".format(len(strain_collection)))

    # ======================= Alignments
    cache = ReadStrainCollectionCache(timeseries_reads, db, strain_collection)
    logger.info("Target cache dir: {}".format(cache.cache_dir))
    error_model = create_error_model(
        observed_reads=timeseries_reads,
        disable_quality=not cfg.model_cfg.use_quality_scores,
        logger=logger
    )
    read_likelihoods = ReadFragmentMappings(
        timeseries_reads, db, error_model,
        cache=cache,
        dtype=cfg.engine_cfg.dtype
    ).model_values

    # ======================= Fit negative binomial distribution.
    avg_marker_len = int(cnp.median([
        len(m)
        for s in db.all_strains()
        for m in s.markers
    ]))
    read_lens = cnp.array([
        len(read)
        for reads_t in timeseries_reads
        for read in reads_t
    ])
    frag_len_negbin_n, frag_len_negbin_p = negbin_fit_frags(avg_marker_len, read_lens, max_padding_ratio=0.5)
    logger.debug("Negative binomial fit: n={}, p={} (mean={}, std={})".format(
        frag_len_negbin_n,
        frag_len_negbin_p,
        frag_len_negbin_n * (1 - frag_len_negbin_p) / frag_len_negbin_p,
        cnp.sqrt(frag_len_negbin_n * (1 - frag_len_negbin_p)) / frag_len_negbin_p
    ))

    for t_idx, reads_t in enumerate(timeseries_reads):
        if len(reads_t) == 0:
            logger.info("Skipping timepoint {} (t_idx={}), because there were zero reads".format(
                reads_t.time_point,
                t_idx
            ))
            continue
        read_likelihoods_t = read_likelihoods.slices[t_idx]

        # Compute fragment frequencies.
        FragmentFrequencyComputer(
            frag_nbinom_n=frag_len_negbin_n,
            frag_nbinom_p=frag_len_negbin_p,
            cache=cache,
            fragments=read_likelihoods_t.fragments,
            fragment_pairs=read_likelihoods_t.fragment_pairs,
            time_idx=t_idx,
            dtype=cfg.engine_cfg.dtype,
            n_threads=cfg.model_cfg.num_cores
        ).get_frequencies()
    logger.info("Done.")


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    main_logger = create_logger("chronostrain.MAIN")
    try:
        main()
    except Exception as e:
        main_logger.exception(e)
        exit(1)
