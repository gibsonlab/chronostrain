import shutil
from typing import Tuple, List
from pathlib import Path
from logging import Logger
import json

from Bio import SeqIO
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from chronostrain.config import ChronostrainConfig
from chronostrain.database import JSONParser, StrainDatabase
from chronostrain.util.external import dashing2_sketch


def prune_json_db_jaccard(
    src_json_path: Path,
    tgt_json_path: Path,
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
    strain_id_ordering, distances = compute_jaccard_distances(src_db, tmp_dir)
    np.save(str(tgt_json_path.parent / "distances.npy"), distances)
    with open(tgt_json_path.parent / "distance_order.txt", "wt") as f:
        for s_id in strain_id_ordering:
            print(s_id, file=f)

    logger.info("Computing clusters.")
    # remove infinities.
    distances[np.isinf(distances)] = 1.0
    clusters, cluster_reps = cluster_db(distances, identity_threshold)

    # ======== Create the clustered json.
    src_strain_ids = set(s.id for s in src_db.all_strains())  # some strains can't be parsed due to IUPAC code/`N`-related restrictions.

    # Parse source JSON.
    with open(src_json_path, "r") as f:
        src_entries = {
            strain_entry['id']: strain_entry
            for strain_entry in json.load(f)
            if strain_entry['id'] in src_strain_ids
        }

    # ======== Move JSON entry from src to target for each cluster; add metadata.
    result_entries = []
    for cluster, rep in zip(clusters, cluster_reps):
        rep_strain_idx = cluster[rep]
        rep_strain_id = strain_id_ordering[rep_strain_idx]

        cluster_entry = src_entries[rep_strain_id]
        cluster_entry['cluster'] = [
            "{}({})".format(
                strain_id_ordering[s_idx],
                src_entries[strain_id_ordering[s_idx]]['name']
            )
            for s_idx in cluster
        ]
        result_entries.append(cluster_entry)

    with open(tgt_json_path, 'w') as outfile:
        json.dump(result_entries, outfile, indent=4)

    logger.info("Before clustering: {} strains".format(len(src_entries)))
    logger.info("After clustering: {} strains".format(len(result_entries)))

    logger.debug("Cleaning up. Removing tmpdir = {}".format(tmp_dir))
    shutil.rmtree(tmp_dir)


def compute_jaccard_distances(
        src_db: StrainDatabase,
        tmp_dir: Path
) -> Tuple[List[str], np.ndarray]:

    # Create Fasta entries.
    input_file_list = tmp_dir / "all_entries.txt"
    with open(input_file_list, 'wt') as input_list_f:
        accs = []
        for strain in src_db.all_strains():
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
        sketch_full_with_multiplicity=True,
        prob_min_hash=True,
        comparison_use_mash_distances=True,
        sketch_size=4096,
        comparison_binary_output=True,  # Flip this to False to debug.
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
        linkage='average',
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
