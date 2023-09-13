from typing import Dict, List
from pathlib import Path
from logging import Logger
from collections import defaultdict
import itertools
import math
import json

import numpy as np
from Bio import SeqIO
from sklearn.cluster import AgglomerativeClustering
from chronostrain.util.sequences.z4 import nucleotides_to_z4


def prune_json_db_multiple_alignments(
        raw_json_path: Path,
        raw_strain_ids: List[str],
        tgt_json_path: Path,
        align_path: Path, logger: Logger,
        identity_threshold: float
):
    # DEPRECATED! use the multiplicity-aware Jaccard index instead.
    # parse json entries.
    raw_strain_id_set = set(raw_strain_ids)
    with open(raw_json_path, "r") as f:
        raw_strain_entries = json.load(f)
        entries = {
            strain_entry['id']: strain_entry
            for strain_entry in raw_strain_entries
            if strain_entry['id'] in raw_strain_id_set
        }

    # perform clustering.
    import numpy as np
    clusters, cluster_reps, distances = cluster_db(
        raw_strain_ids, raw_strain_entries, align_path, logger, ident_fraction=identity_threshold
    )
    np.save(str(tgt_json_path.parent / "distances.npy"), distances)

    # Create the clustered json.
    result_entries = []
    for cluster, rep in zip(clusters, cluster_reps):
        rep_strain_idx = cluster[rep]
        rep_strain = raw_strain_ids[rep_strain_idx]

        cluster_entry = entries[rep_strain]
        cluster_entry['cluster'] = [
            "{}({})".format(raw_strain_ids[s_idx], entries[raw_strain_ids[s_idx]]['name'])
            for s_idx in cluster
        ]
        result_entries.append(cluster_entry)

    with open(tgt_json_path, 'w') as outfile:
        json.dump(result_entries, outfile, indent=4)

    logger.info("Before clustering: {} strains".format(len(entries)))
    logger.info("After clustering: {} strains".format(len(result_entries)))


def cluster_db(
        strain_ids: List[str],
        strain_entries: List[Dict],
        alignments_path: Path,
        ident_fraction: float,  # corresponds to 99.8% seq identity
        logger: Logger
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
        metric='precomputed',
        linkage='average',
        distance_threshold=math.ceil((1 - ident_fraction) * align_len),
        n_clusters=None
    ).fit(distances)

    n_clusters, cluster_labels = clustering.n_clusters_, clustering.labels_
    clusters: List[List[int]] = [
        [s_idx for s_idx in np.where(cluster_labels == c)[0]]
        for c in range(n_clusters)
    ]

    logger.info("Initial clustering: {} Clusters".format(len(clusters)))

    # Ensure members of clusters have identical copy # of each gene.
    copy_number_ensured_clusters: List[List[int]] = []
    for cluster in clusters:
        copy_number_ensured_clusters += split_by_gene_copy_number(cluster, strain_entries)
    clusters = copy_number_ensured_clusters
    logger.info("After copy number separation: {} Clusters".format(len(clusters)))

    cluster_reps = pick_cluster_representatives(clusters, distances)
    return clusters, cluster_reps, distances


def split_by_gene_copy_number(cluster: List[int], strain_entries: List[Dict]) -> List[List[int]]:
    # preprocessing: gather the possible genes.
    gene_names = set()
    for strain_entry in strain_entries:
        for marker_entry in strain_entry['markers']:
            gene_names.add(marker_entry['name'])
    gene_names = list(gene_names)

    # Tally up genes by name.
    member_signatures = {}
    for s_idx in cluster:
        strain_entry = strain_entries[s_idx]
        gene_counts = {g: 0 for g in gene_names}
        for marker_entry in strain_entry['markers']:
            gene_name = marker_entry['name']
            gene_counts[gene_name] += 1
        member_signatures[s_idx] = "|".join(str(gene_counts[g]) for g in gene_names)

    # group members by signature.
    signature_components = defaultdict(list)
    for s_idx, signature in member_signatures.items():
        signature_components[signature].append(s_idx)

    # Output.
    return [
        components
        for signature, components in signature_components.items()
    ]


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
