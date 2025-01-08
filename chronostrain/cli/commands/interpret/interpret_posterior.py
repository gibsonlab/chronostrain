from typing import Union

import click
from pathlib import Path

from ..base import option


@click.command()
@option(
    '--advi-outdir', '-a', 'advi_outdir',
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, exists=True),
    required=True,
    help="The path to the model parameters output by the ADVI inference command. "
         "This should look like `posterior.<dtype>.npz`."
)
@option(
    '--filtered-reads', '-r', 'filtered_reads',
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the filtered read CSV/TSV file used as input for abundance estimation."
)
@option(
    '--outdir', '-o', 'out_dir',
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    required=True,
    help="The directory to output the abundance profiles to."
)
@option(
    '--strain-subset', '-s', 'strain_subset_path',
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False, exists=True, readable=True),
    required=False, default=None,
    help="A text file specifying a subset of database strain IDs (the -s option in ADVI) used for clustering."
)
@option(
    '--n-samples', '-n', 'n_samples',
    type=int, required=False, default=5000,
    help="The number of posterior samples to use."
)
@option(
    '--posterior-threshold', '-p', 'posterior_threshold',
    type=float, required=False, default=0.95,
    help="The posterior threshold to use for filtering the output. All strains with posterior q(Z_s) less than this "
         "value will be zeroed out prior to renormalization."
)
@option(
    '--restrict-species', '-rs', 'restrict_species_name',
    type=str, required=False, default="",
    help="The species to restrict the output to, in the format \"GENUS SPECIES\" (Example: -r \"Escherichia coli\"), case insensitive."
         "This command will automatically renormalize on this subset of strains. "
         "This command is useful for the (typical) scenario where an entire family has been included for analysis, "
         "but the profiling was meant for a particular species."
)
@option(
    '--convert-to-overall/--dont-convert-to-overall', 'convert_to_overall',
    is_flag=True, default=False,
    help="If specified, converts the database-relative abundance ratios to the sample-overall relative abundances. "
         "This requires that all genomes being profiled (e.g. the -r and -s arguments) must have a whole-genome "
         "length estimate in the JSON database entry. "
)
def main(
        advi_outdir: Path,
        filtered_reads: Path,
        out_dir: Path,
        strain_subset_path: Path,
        n_samples: int,
        posterior_threshold: float,
        restrict_species_name: Union[str, None],
        convert_to_overall: bool
):
    """
    Convert ADVI raw output into an ensemble of abundance profiles.
    """
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli.interpret_posterior")

    import numpy as np
    from chronostrain.config import cfg
    from chronostrain.model import TimeSeriesReads
    from chronostrain.inference import AbstractReparametrizedPosterior, ReparametrizedGaussianPosterior, GaussianWithGumbelsPosterior
    from .helpers.interpret_helper import interpret_posterior_with_zeroes, parse_strains, parse_adhoc_clusters, convert_relative_to_overall

    db = cfg.database_cfg.get_database()
    if strain_subset_path is not None:
        with open(strain_subset_path, "rt") as f:
            strains_to_profile = [db.get_strain(line.strip().split('\t')[0]) for line in f if not line.startswith("#")]
        logger.info("Loaded list of {} strains.".format(len(strains_to_profile)))
    else:
        strains_to_profile = db.all_strains()
        logger.info("Using complete collection of {} strains from database.".format(len(strains_to_profile)))

    if len(restrict_species_name) > 0:
        taxa_tokens = restrict_species_name.split(" ")
        if len(taxa_tokens) > 2:
            raise ValueError("restrict_species argument should be two words separated by a space.")
        target_genus, target_species = taxa_tokens
        strains_to_profile = [
            s for s in strains_to_profile
            if (s.metadata.genus.lower() == target_genus.lower() and s.metadata.species.lower() == target_species.lower())
        ]
        logger.info("Restricting to target species {} {} (found {} strain entries/clusters)".format(
            target_genus, target_species, len(strains_to_profile)
        ))

    filt_reads = TimeSeriesReads.load_from_file(filtered_reads)
    time_points = np.array([reads_t.time_point for reads_t in filt_reads], dtype=float)

    post_inference_strains = parse_strains(db, advi_outdir / "strains.txt")
    try:
        adhoc_clustering = parse_adhoc_clusters(db, advi_outdir / "adhoc_cluster.txt")
    except FileNotFoundError:
        logger.info("No adhoc_cluster.txt found. This is expected if inference was run without model trimming (an atypical scenario).")
        adhoc_clustering = {s.id: s for s in db.all_strains()}
    posterior = AbstractReparametrizedPosterior.load_class_from_initializer(advi_outdir / "posterior.{}.metadata".format(cfg.engine_cfg.dtype))
    posterior.load(advi_outdir / "posterior.{}.npz".format(cfg.engine_cfg.dtype))
    out_dir.mkdir(exist_ok=True, parents=True)
    if isinstance(posterior, GaussianWithGumbelsPosterior):
        logger.info("Using posterior threshold = {}".format(posterior_threshold))
        posterior_ratios = interpret_posterior_with_zeroes(
            logger,
            posterior,
            n_samples,
            posterior_threshold,
            strains_to_profile,
            post_inference_strains,
            adhoc_clustering
        )

        if convert_to_overall:
            num_filtered_reads = [len(reads_t) for reads_t in filt_reads]
            read_depths = [reads_t.read_depth for reads_t in filt_reads]

            abundance_profile = convert_relative_to_overall(
                posterior_ratios,
                strains_to_profile,
                read_depths,
                num_filtered_reads,
            )
        else:
            abundance_profile = posterior_ratios

        with open(out_dir / "profiled_strains.txt", "wt") as f:
            for strain in strains_to_profile:
                f.write(f"{strain.id}\n")
        np.save(out_dir / "abundance_profile.npy", abundance_profile)
        np.save(out_dir / "time_points.npy", time_points)
        logger.info("Finished the conversion.")
    elif isinstance(posterior, ReparametrizedGaussianPosterior):
        raise NotImplementedError("[todo] implement posterior interpretation for simple model without q(Z). Note: posterior_threshold should have no effect for this.")
