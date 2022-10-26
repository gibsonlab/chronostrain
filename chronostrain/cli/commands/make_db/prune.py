from typing import Dict, Any, List
from pathlib import Path
from logging import Logger
import json
import itertools
import math

import numpy as np
from Bio import SeqIO
from sklearn.cluster import AgglomerativeClustering

from chronostrain.util.sequences import nucleotides_to_z4


def prune_db(input_json_path: Path, output_json_path: Path, alignments_path: Path, logger: Logger):
    logger.info("Preprocessing for pruning.")

    # Read the alignments.
    alignments: Dict[str, np.ndarray] = {}
    align_len = 0
    for record in SeqIO.parse(alignments_path, "fasta"):
        accession = record.id
        alignments[accession] = nucleotides_to_z4(str(record.seq))
        align_len = len(record.seq)

    # parse json entries.
    entries: Dict[str, Dict[str, Any]] = {}
    strain_ids = []
    with open(input_json_path, "r") as f:
        _initial_strain_entries = json.load(f)
        for strain_entry in _initial_strain_entries:
            accession = strain_entry['id']
            if accession not in alignments:
                logger.warning(
                    f"Strain `{accession} not found in the multiple alignments. "
                    f"This is to be expected, if it wasn't parseable. "
                    f"Check the logs which first loaded the database."
                )
                continue
            entries[accession] = strain_entry
            strain_ids.append(accession)

    logger.info("Computing distances.")
    distances = np.zeros(shape=(len(strain_ids), len(strain_ids)), dtype=int)
    for (i1, strain1_id), (i2, strain2_id) in itertools.combinations(enumerate(strain_ids), r=2):
        hamming_dist = np.sum(alignments[strain1_id] != alignments[strain2_id])
        distances[i1, i2] = hamming_dist
        distances[i2, i1] = hamming_dist

    logger.info("Computing clusters.")
    ident_fraction = 0.01  # corresponds to 1% identity
    clustering = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        distance_threshold=math.ceil(ident_fraction * align_len),
        n_clusters=None
    ).fit(distances)

    n_clusters, cluster_labels = clustering.n_clusters_, clustering.labels_
    clusters: List[List[int]] = [
        [s_idx for s_idx in np.where(cluster_labels == c)[0]]
        for c in range(n_clusters)
    ]

    cluster_reps = pick_cluster_representatives(clusters, distances)

    # Create the clustered json.
    result_entries = []
    for cluster, rep in zip(clusters, cluster_reps):
        rep_strain_idx = cluster[rep]
        rep_strain = strain_ids[rep_strain_idx]

        cluster_entry = entries[rep_strain]
        cluster_entry['cluster'] = [
            "{}({})".format(strain_ids[s_idx], entries[strain_ids[s_idx]]['name'])
            for s_idx in cluster
        ]
        result_entries.append(cluster_entry)

    with open(output_json_path, 'w') as outfile:
        json.dump(result_entries, outfile, indent=4)

    logger.info("Before clustering: {} strains".format(len(entries)))
    logger.info("After clustering: {} strains".format(len(result_entries)))


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
