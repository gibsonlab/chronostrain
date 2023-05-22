from typing import *
from pathlib import Path
from logging import Logger

import numpy as np
import click

from chronostrain.model import StrainMetadata
from chronostrain.database import StrainDatabase, PickleParser
from ..base import option


@click.command()
@click.pass_context
@option(
    '--input-db', '-i', 'input_db_name',
    type=str, required=True, help="The target database file to prune (assumes a pickle exists)."
)
@option(
    '--output-db', '-o', 'output_db_name',
    type=str, required=True, help="The target database file to prune (assumes a pickle exists)."
)
@option(
    '--distance-threshold', '-t', 'distance_threshold',
    type=float, required=False, default=0.0005,
    help="The distance threshold to use for Agglomerative clustering (a fraction between 0 and 1), representing"
         "one minus the percent identity in the concatenated multiple alignment."
)
def main(
        ctx: click.Context,
        input_db_name: str,
        output_db_name: str,
        distance_threshold: float,
):
    """
    Perform posterior estimation using ADVI.
    """
    ctx.ensure_object(Logger)
    logger = ctx.obj

    from chronostrain.config import cfg

    # ==== Initialize database instance.
    logger.info(f"Loading DB instance, using data directory: {cfg.database_cfg.data_dir}")
    input_db = PickleParser(input_db_name, cfg.database_cfg.data_dir).parse()

    # ============== Step 2: prune using multiple alignments.
    logger.info("Pruning database by constructing multiple alignments.")
    from .multiple_alignments import marker_concatenated_multiple_alignments

    marker_names = sorted(input_db.all_marker_names())
    align_path = input_db.work_dir / "multiple_alignment.fasta"
    marker_concatenated_multiple_alignments(input_db, align_path, marker_names, logger)
    prune_db(input_db, output_db_name, align_path, distance_threshold, logger, cfg)


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    my_logger = create_logger("chronostrain.prune_pickle")
    try:
        main(obj=my_logger)
    except Exception as e:
        my_logger.exception(e)
        exit(1)


def prune_db(input_db: StrainDatabase, output_db_name: str, alignments_path: Path, distance_threshold: float, logger: Logger, cfg):
    import itertools
    import math
    from Bio import SeqIO
    from sklearn.cluster import AgglomerativeClustering
    from chronostrain.util.sequences.z4 import nucleotides_to_z4
    from chronostrain.database.backend import PandasAssistedBackend
    from .prune import pick_cluster_representatives

    logger.info("Preprocessing for pruning.")
    # Read the alignments.
    alignments: Dict[str, np.ndarray] = {}
    align_len = 0
    for record in SeqIO.parse(alignments_path, "fasta"):
        accession = record.id
        alignments[accession] = nucleotides_to_z4(str(record.seq))
        align_len = len(record.seq)

    strains = input_db.all_strains()

    logger.info("Computing distances.")
    distances = np.zeros(shape=(len(strains), len(strains)), dtype=int)
    for (i1, strain1), (i2, strain2) in itertools.combinations(enumerate(strains), r=2):
        hamming_dist = np.sum(alignments[strain1.id] != alignments[strain2.id])
        distances[i1, i2] = hamming_dist
        distances[i2, i1] = hamming_dist

    logger.info("Computing clusters.")
    clustering = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        distance_threshold=math.ceil(distance_threshold * align_len),
        n_clusters=None
    ).fit(distances)

    # noinspection PyUnresolvedReferences
    n_clusters, cluster_labels = clustering.n_clusters_, clustering.labels_
    clusters: List[List[int]] = [
        [s_idx for s_idx in np.where(cluster_labels == c)[0]]
        for c in range(n_clusters)
    ]

    cluster_reps = pick_cluster_representatives(clusters, distances)

    # Create the clustered database.
    new_strains = []
    for cluster, rep in zip(clusters, cluster_reps):
        rep_strain_idx = cluster[rep]
        rep_strain = strains[rep_strain_idx]
        if rep_strain.metadata is None:
            # noinspection PyTypeChecker
            rep_strain.metadata = StrainMetadata(
                None,
                None,
                None,
                None,
                None,
                [strains[s_idx].id for s_idx in cluster]
            )
        new_strains.append(rep_strain)

    new_db = StrainDatabase(
        backend=PandasAssistedBackend(),
        data_dir=cfg.database_cfg.data_dir,
        name=output_db_name,
    )
    serializer = PickleParser(output_db_name, cfg.database_cfg.data_dir)

    new_db.backend.add_strains(new_strains)
    new_db.save_markers_to_multifasta(force_refresh=True)
    serializer.save_to_disk(new_db)

    logger.info("Before clustering: {} strains".format(input_db.num_strains()))
    logger.info("After clustering: {} strains".format(new_db.num_strains()))
    logger.info("Successfully created new database instance `{}`.".format(output_db_name))
    logger.info("Pickle path: {}".format(serializer.pickle_path()))
