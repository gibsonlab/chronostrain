import shutil
from typing import Tuple, List
from pathlib import Path
from logging import Logger

from Bio import SeqIO
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from chronostrain.config import ChronostrainConfig
from chronostrain.database import JSONParser, StrainDatabase
from chronostrain.util.external import dashing2_sketch


def cluster_json_db_jaccard(
    src_json_path: Path,
    output_path: Path,
    tmp_dir: Path,
    cfg: ChronostrainConfig,
    identity_threshold: float,
    logger: Logger
):
    logger.info(f"Preprocessing step -- Loading old DB instance, using data directory: {cfg.database_cfg.data_dir}")
    src_db = JSONParser(
        entries_file=src_json_path,
        data_dir=cfg.database_cfg.data_dir,
        marker_max_len=cfg.database_cfg.parser_kwargs['marker_max_len'],
        force_refresh=False
    ).parse()

    # ======== Do the calculations.
    tmp_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Computing all-to-all Jaccard distance calculations.")
    #sketch_mode = "FullCountDict"
    sketch_mode = "ProbMinHash"
    sketch_size = 4096 * 2
    logger.info(f"Using {sketch_mode} method for sketching (sz={sketch_size}).")
    strain_id_ordering, distances = compute_jaccard_distances(
        src_db, logger, tmp_dir,
        dashing2_sketch_size=sketch_size,
        sketch_mode=sketch_mode
    )
    np.save(str(output_path.parent / "distances.npy"), distances)
    with open(output_path.parent / "distance_order.txt", "wt") as f:
        for s_id in strain_id_ordering:
            print(s_id, file=f)

    logger.info("Computing clusters.")
    # remove infinities.
    distances[np.isinf(distances)] = 1.0
    clusters, cluster_reps = cluster_db(distances, identity_threshold)

    with open(output_path, 'w') as outfile:
        print("# Clustering performed at {} ident".format(identity_threshold), file=outfile)
        for cluster, rep in zip(clusters, cluster_reps):
            rep_strain_idx = cluster[rep]
            rep_strain_id = strain_id_ordering[rep_strain_idx]
            cluster_strain_ids = [
                strain_id_ordering[s_idx]
                for s_idx in cluster
            ]

            print(
                "{}\t{}".format(
                    rep_strain_id,
                    ','.join(cluster_strain_ids)
                ),
                file=outfile
            )

    logger.info("Before clustering: {} genomes".format(src_db.num_strains()))
    logger.info("After clustering: {} genomes".format(len(clusters)))

    logger.debug("Cleaning up. Removing tmpdir = {}".format(tmp_dir))
    shutil.rmtree(tmp_dir)


def compute_jaccard_distances(
        src_db: StrainDatabase,
        logger: Logger,
        tmp_dir: Path,
        dashing2_sketch_size: int = 4096,
        sketch_mode: str = "ProbMinHash"
) -> Tuple[List[str], np.ndarray]:

    # Create Fasta entries.
    input_file_list = tmp_dir / "all_entries.txt"
    with open(input_file_list, 'wt') as input_list_f:
        accs = []
        for strain in src_db.all_strains():
            num_markers_with_N = 0
            for m in strain.markers:
                if m.seq.number_of_ns() > 0:
                    logger.warning(f"Strain {strain.id} ({strain.metadata.genus} {strain.metadata.species}, {strain.name}) -- marker {m.name} has an N. This strain will be excluded.")
                    num_markers_with_N += 1
            if num_markers_with_N > 0:
                continue

            strain_marker_fasta = tmp_dir / f'{strain.id}.fasta'
            accs.append(strain.id)
            with open(strain_marker_fasta, 'wt') as out_f:
                for marker in strain.markers:
                    SeqIO.write(
                        [marker.to_seqrecord()],
                        out_f, 'fasta'
                    )
            print(str(strain_marker_fasta), file=input_list_f)

    # Invoke dashing2.
    sketch_file = tmp_dir / 'sketches.dat'
    distance_file = tmp_dir / "distances.dat"

    dashing2_sketch(
        filename_list_input=input_file_list,
        comparison_out=distance_file,
        sketch_outfile=sketch_file,
        min_hash_mode=sketch_mode,
        emit_distances=True,
        sketch_size=dashing2_sketch_size,
        binary_output=True,  # Flip this to False to debug.
        silent=True
    )

    n = len(accs)
    """Use the memmap helper, as seen in example: https://github.com/dnbaker/dashing2/blob/main/python/parse.py"""
    raw_values = np.memmap(distance_file, np.float32)
    if len(raw_values) != int(n * (n-1) * 0.5):
        raise ValueError("Expected upper triangular distance matrix, but the numer of entries didn't match.")

    triu_distances = np.zeros((n, n), dtype=float)
    triu_distances[np.triu_indices(n=n, m=n, k=1)] = raw_values
    del raw_values
    return accs, triu_distances + triu_distances.T


def cluster_db(
        distances: np.ndarray,
        identity_threshold: float
) -> Tuple[
    List[List[int]],
    List[int]
]:
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='complete',
        distance_threshold=(1 - identity_threshold),
        n_clusters=None
    ).fit(distances)

    n_clusters, cluster_labels = clustering.n_clusters_, clustering.labels_
    clusters: List[List[int]] = [
        [s_idx for s_idx in np.where(cluster_labels == c)[0]]
        for c in range(n_clusters)
    ]

    cluster_reps = pick_cluster_representatives(clusters, distances)
    return clusters, cluster_reps


def pick_cluster_representatives(clusters: List[List[int]], distances: np.ndarray) -> List[int]:
    """
    Decide the cluster reps by looking for the node that most closely resembles the cluster-wide average distances.
    """
    reps = []
    for c_idx, cluster in enumerate(clusters):
        cluster_averages = []
        node_averages = []
        for other_c_idx, other_cluster in enumerate(clusters):
            # (include the same cluster in this calculation.)
            submatrix = distances[np.ix_(cluster, other_cluster)]
            cluster_averages.append(np.mean(submatrix))
            node_averages.append(np.mean(submatrix, axis=1))

        cluster_averages = np.array(cluster_averages)
        node_averages = np.stack(node_averages, axis=1)

        # Difference form cluster-wide averages.
        differences = node_averages - cluster_averages.reshape(1, -1)
        rep = int(np.argmin(
            np.abs(differences).sum(axis=1)  # Minimize L1-norm of the difference vector.
        ))

        # Note: this indexing is relative to each cluster (e.g. "0" is the first element of the cluster).
        reps.append(rep)
    return reps
