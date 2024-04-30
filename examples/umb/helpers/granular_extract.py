from typing import List, Dict, Tuple, Set

import click
from pathlib import Path
from click import option

import numpy as np
import scipy.special
from chronostrain.database import StrainDatabase
from chronostrain.model import Strain, TimeSeriesReads
from chronostrain.inference import GaussianWithGumbelsPosterior
from chronostrain.config import cfg


@click.command()
@option(
    '--reads', '-r', 'reads_input',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the reads input CSV file."
)
@option(
    '--coarse-dir', '-c', 'coarse_inference_outdir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="The directory that contains the result of the ``coarse'' inference."
)
@option(
    '--coarse-clustering', '-cc', 'coarse_clustering_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="The directory that contains the result of the ``coarse'' inference."
)
@option(
    '--granular-clustering', '-g', 'granular_id_file',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="The JSON file that points to the desired granular database specification."
)
@option(
    '--out', '-o', 'out_path',
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="The path to write the target ID list to."
)
@option(
    '--prior-p', 'prior_p', type=float, default=0.5,
    help='The prior bias for the indicator variables, where bias = P(strain included in model).'
)
@option(
    '--abund-lb', '-lb', 'abund_lb', type=float, default=0.05,
    help='The (database-normalized) abund lower bound to determine presence/absence from sample. '
         'Only used for parsing the coarse output.'
)
@option(
    '--bayes-factor', '-bf', 'bf_threshold', type=float, default=100000.0,
    help='The Bayes factor threshold for parsing the coarse output.'
)
def main(
        reads_input: Path,
        coarse_inference_outdir: Path,
        coarse_clustering_path: Path,
        granular_id_file: Path,
        out_path: Path,
        prior_p: float,
        abund_lb: float,
        bf_threshold: float
):
    """
    Perform posterior estimation using ADVI.
    """
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli.granular_extract")
    from chronostrain.config import cfg
    from chronostrain.inference import GaussianStrainCorrelatedWithGlobalZerosPosterior

    # ============ Extract the coarse inference results.
    coarse_db = cfg.database_cfg.get_database()
    coarse_clustering = load_chronostrain_cluster(coarse_clustering_path)
    coarse_inference_strain_ids_full = extract_coarse_inference(
        filt_reads_path=reads_input,
        coarse_inference_outdir=coarse_inference_outdir,
        clustering=coarse_clustering,
        coarse_db=coarse_db,
        posterior_class=GaussianStrainCorrelatedWithGlobalZerosPosterior,
        bf_threshold=bf_threshold,
        prior_p=prior_p,
        abund_lb=abund_lb
    )
    logger.info("Coarse inference got {} strain IDs after unwrapping each cluster.".format(
        len(coarse_inference_strain_ids_full)
    ))

    # ============ Create database instance.
    with open(granular_id_file, "rt") as f:
        granular_strain_ids: Set[str] = {
            line.strip().split('\t')[0]
            for line in f
            if not line.startswith("#")
        }

    coarse_inference_strain_ids = coarse_inference_strain_ids_full.intersection(granular_strain_ids)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "wt") as f:
        print(f"# Granular inference IDs extracted from {coarse_inference_outdir}", file=f)
        for s_id in coarse_inference_strain_ids:
            print(s_id, file=f)


def parse_adhoc_clusters(db: StrainDatabase, txt_file: Path) -> Dict[str, Strain]:
    clust = {}
    with open(txt_file, "rt") as f:
        for line in f:
            tokens = line.strip().split(":")
            rep = tokens[0]
            members = tokens[1].split(",")
            for member in members:
                clust[member] = db.get_strain(rep)
    return clust


def parse_strains(db: StrainDatabase, strain_txt: Path):
    with open(strain_txt, 'rt') as f:
        return [
            db.get_strain(l.strip())
            for l in f
        ]


def load_chronostrain_cluster(chronostrain_cluster: Path) -> Dict[str, List[str]]:
    clustering = {}
    with open(chronostrain_cluster, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            tokens = line.strip().split("\t")
            rep_id = tokens[0]
            members = list(tokens[1].split(","))
            clustering[rep_id] = members
    return clustering


def extract_coarse_inference(
        filt_reads_path: Path,
        coarse_inference_outdir: Path,
        clustering: Dict[str, List[str]],
        coarse_db: StrainDatabase,
        posterior_class,
        bf_threshold: float,
        prior_p: float,
        abund_lb: float
) -> Set[str]:
    """
    Apply the detection filters to the coarse inference posterior, and output all strain IDs found this way.
    In addition, unwind each cluster to its members: if the filter gives 5 clusters with 3 members per cluster, then
    the resulting list contains 15 IDs.
    """
    adhoc_clusters: Dict[str, Strain] = parse_adhoc_clusters(coarse_db, coarse_inference_outdir / "adhoc_cluster.txt")
    inference_strains: List[Strain] = parse_strains(coarse_db, coarse_inference_outdir / 'strains.txt')
    display_strains: List[Strain] = list(coarse_db.get_strain(x) for x in adhoc_clusters.keys())

    reads = TimeSeriesReads.load_from_file(filt_reads_path)
    time_points = np.array([reads_t.time_point for reads_t in reads], dtype=float)
    posterior = posterior_class(
        len(inference_strains),
        len(time_points),
        cfg.engine_cfg.dtype
    )
    posterior.load(Path(coarse_inference_outdir / "posterior.{}.npz".format(cfg.engine_cfg.dtype)))

    posterior_p, posterior_samples = posterior_with_bf_threshold(
        posterior=posterior,
        inference_strains=inference_strains,
        output_strains=display_strains,
        adhoc_clustering=adhoc_clusters,
        bf_threshold=bf_threshold,
        prior_p=prior_p
    )

    strains_to_output = {}
    for t_idx, t in enumerate(time_points):
        for strain_idx, strain in enumerate(display_strains):
            filt_relabunds = posterior_samples[t_idx, :, strain_idx]
            if np.median(filt_relabunds) > abund_lb:
                strains_to_output[strain.id] = strain

    strain_full_ids = set()
    for s_id, strain in strains_to_output.items():
        for member_id in clustering[s_id]:
            strain_full_ids.add(member_id)
    return strain_full_ids


def posterior_with_bf_threshold(
        posterior: GaussianWithGumbelsPosterior,
        inference_strains: List[Strain],
        output_strains: List[Strain],
        adhoc_clustering: Dict[str, Strain],
        bf_threshold: float,
        prior_p: float
) -> Tuple[Dict[str, float], np.ndarray]:
    # Raw random samples.
    n_samples = 5000
    rand = posterior.random_sample(n_samples)
    g_samples = np.array(
        posterior.reparametrized_gaussians(rand['std_gaussians'], posterior.get_parameters()))  # T x N x S
    z_samples = np.array(posterior.reparametrized_zeros(rand['std_gumbels'], posterior.get_parameters()))
    # print(posterior.get_parameters())# N x S

    n_times = g_samples.shape[0]
    n_inference_strains = g_samples.shape[-1]
    assert n_inference_strains == len(inference_strains)

    # Calculate bayes factors.
    posterior_inclusion_p = scipy.special.expit(-posterior.get_parameters()['gumbel_diff'])
    # print(posterior_inclusion_p)
    posterior_inclusion_bf = (posterior_inclusion_p / (1 - posterior_inclusion_p)) * ((1 - prior_p) / prior_p)

    # Calculate abundance estimates using BF thresholds.
    indicators = np.full(n_inference_strains, fill_value=False, dtype=bool)
    indicators[posterior_inclusion_bf > bf_threshold] = True
    print("{} of {} inference strains passed BF Threshold > {}".format(np.sum(indicators), n_inference_strains,
                                                                       bf_threshold))

    log_indicators = np.empty(n_inference_strains, dtype=float)
    log_indicators[indicators] = 0.0
    log_indicators[~indicators] = -np.inf
    pred_abundances_raw = scipy.special.softmax(g_samples + np.expand_dims(log_indicators, axis=[0, 1]), axis=-1)

    # Unwind the adhoc grouping.
    pred_abundances = np.zeros(shape=(n_times, n_samples, len(output_strains)), dtype=float)
    adhoc_indices = {s.id: i for i, s in enumerate(inference_strains)}
    output_indices = {s.id for s in output_strains}
    for s_idx, s in enumerate(output_strains):
        adhoc_rep = adhoc_clustering[s.id]
        adhoc_idx = adhoc_indices[adhoc_rep.id]
        adhoc_clust_ids = set(s_ for s_, clust in adhoc_clustering.items() if clust.id == adhoc_rep.id)
        adhoc_sz = len(adhoc_clust_ids.intersection(output_indices))
        # if adhoc_sz > 1:
        #     print(f"{s.id} [{s.metadata.genus} {s.metadata.species}, {s.name}] --> adhoc sz = {adhoc_sz} (Adhoc Cluster {adhoc_rep.id} [{adhoc_rep.metadata.genus} {adhoc_rep.metadata.species}, {adhoc_rep.name}])")
        pred_abundances[:, :, s_idx] = pred_abundances_raw[:, :, adhoc_idx] / adhoc_sz
    return {
        s.id: posterior_inclusion_p[
            adhoc_indices[adhoc_clustering[s.id].id]
        ]
        for i, s in enumerate(output_strains)
    }, pred_abundances


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    main_logger = create_logger("chronostrain.MAIN")
    try:
        main()
    except Exception as e:
        main_logger.exception(e)
        exit(1)
