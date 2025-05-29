from logging import Logger
from pathlib import Path
from typing import *

import numpy as np
import scipy.special

from chronostrain.database import StrainDatabase
from chronostrain.model import Strain
from chronostrain.inference import GaussianWithGumbelsPosterior


def parse_strains(db: StrainDatabase, strain_txt: Path):
    with open(strain_txt, 'rt') as f:
        return [
            db.get_strain(l.strip())
            for l in f
        ]


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


def total_marker_len(strain: Strain) -> int:
    return sum(len(m) for m in strain.markers)


def convert_relative_to_overall(
        abundance_ratios: np.ndarray,
        strains: List[Strain],
        read_depths: List[int],
        filtered_reads: List[int],
) -> np.ndarray:
    marker_lens = np.array([
        total_marker_len(strain) for strain in strains
    ], dtype=int)
    genome_lens = np.array([
        strain.metadata.total_len for strain in strains
    ], dtype=int)

    invalid_genome_len_counts = np.sum(genome_lens == 0)
    if invalid_genome_len_counts > 0:
        raise ValueError("The following genomes do not have a specified genome length: {}".format(
            ",".join(
                s.id
                for s, l in zip(strains, genome_lens)
                if l == 0
            )
        ))

    marker_ratio = np.reciprocal(np.sum(
        np.expand_dims(marker_lens / genome_lens, axis=[0, 1]) * abundance_ratios,
        axis=-1
    ))  # (T x N)
    read_ratio = np.array(filtered_reads) / np.array(read_depths)  # length T
    weights = marker_ratio * np.expand_dims(read_ratio, axis=1)  # (T x N)
    return abundance_ratios * np.expand_dims(weights, axis=2)


def interpret_posterior_with_zeroes(
        logger: Logger,
        posterior: GaussianWithGumbelsPosterior,
        n_samples: int,
        posterior_threshold: float,
        strains_to_profile: List[Strain],
        post_inference_strains: List[Strain],
        adhoc_clustering: Dict[str, Strain],
) -> Tuple[np.ndarray, np.ndarray]:
    rand = posterior.random_sample(n_samples)
    g_samples = np.array(posterior.reparametrized_gaussians(rand['std_gaussians'], posterior.get_parameters()))  # T x N x S
    # z_samples = np.array(posterior.reparametrized_zeros(rand['std_gumbels'], posterior.get_parameters()))

    n_times = g_samples.shape[0]
    n_inference_strains = g_samples.shape[-1]
    assert n_inference_strains == len(post_inference_strains)

    # Filter by posterior probability.
    posterior_inclusion_p = scipy.special.expit(-posterior.get_parameters()['gumbel_diff'])
    indicators = np.full(n_inference_strains, fill_value=False, dtype=bool)
    indicators[posterior_inclusion_p > posterior_threshold] = True
    logger.info("{} of {} inference strains passed Posterior p(Z_s|Data) > {}".format(np.sum(indicators), n_inference_strains, posterior_threshold))

    log_indicators = np.empty(n_inference_strains, dtype=float)
    log_indicators[indicators] = 0.0
    log_indicators[~indicators] = -np.inf
    pred_abundances_raw = scipy.special.softmax(g_samples + np.expand_dims(log_indicators, axis=[0, 1]), axis=-1)

    # Unwind the adhoc grouping.
    pred_abundances = np.zeros(shape=(n_times, n_samples, len(strains_to_profile)), dtype=float)
    full_posterior_inclusion_p = np.zeros(len(strains_to_profile), dtype=float)  # this is the non-adhoc-clustered version of posterior_inclusion_p

    adhoc_indices = {s.id: i for i, s in enumerate(post_inference_strains)}
    output_indices = {s.id for s in strains_to_profile}
    for s_idx, s in enumerate(strains_to_profile):
        if s.id in adhoc_clustering:
            adhoc_rep = adhoc_clustering[s.id]
            adhoc_idx = adhoc_indices[adhoc_rep.id]
            adhoc_clust_ids = set(s_ for s_, clust in adhoc_clustering.items() if clust.id == adhoc_rep.id)
            adhoc_sz = len(adhoc_clust_ids.intersection(output_indices))
            pred_abundances[:, :, s_idx] = pred_abundances_raw[:, :, adhoc_idx] / adhoc_sz
            full_posterior_inclusion_p[s_idx] = posterior_inclusion_p[adhoc_idx]
        else:
            pred_abundances[:, :, s_idx] = 0.0
            full_posterior_inclusion_p[s_idx] = 0.0

    return pred_abundances, full_posterior_inclusion_p
