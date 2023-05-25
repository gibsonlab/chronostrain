from typing import Dict, List
from pathlib import Path
from logging import Logger
import itertools
import math

import numpy as np
from Bio import SeqIO
from sklearn.cluster import AgglomerativeClustering
from chronostrain.util.sequences.z4 import nucleotides_to_z4


def cluster_db(
        strain_ids: List[str],
        alignments_path: Path,
        logger: Logger,
        ident_fraction: float = 0.002  # corresponds to 99.8% seq identity
):
    logger.info("Performing clustering using multiple alignments.")

    # Read the alignments.
    alignments: Dict[str, np.ndarray] = {}
    align_len = 0
    for record in SeqIO.parse(alignments_path, "fasta"):
        accession = record.id
        alignments[accession] = nucleotides_to_z4(str(record.seq))
        align_len = len(record.seq)

    logger.info("Computing distances.")
    distances = np.zeros(shape=(len(strain_ids), len(strain_ids)), dtype=int)
    for (i1, strain1_id), (i2, strain2_id) in itertools.combinations(enumerate(strain_ids), r=2):
        hamming_dist = np.sum(alignments[strain1_id] != alignments[strain2_id])
        distances[i1, i2] = hamming_dist
        distances[i2, i1] = hamming_dist

    logger.info("Computing clusters.")
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
