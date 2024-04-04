from typing import *
from pathlib import Path

import jax.numpy as jnp

from chronostrain.config import cfg
from chronostrain.inference.likelihoods.read_fragment_mappings import TimeSliceLikelihoods
from chronostrain.model import TimeSeriesReads, TimeSliceReads, FragmentFrequencySparse
from chronostrain.util.math import log_spspmm_exp

from chronostrain.inference.likelihoods import FragmentFrequencyComputer, TimeSeriesLikelihoods, \
    ReadStrainCollectionCache
from .util import divide_columns_into_batches_sparse

from chronostrain.logging import create_logger
logger = create_logger(__name__)


_metadata_file = "batch_metadata.txt"


def load_batched_matrices(matrix_dir: Path) -> List[jnp.ndarray]:
    """
    Implements the load() functionality for cache.
    """
    logger.debug("Loading batched matrices from subdir: {}".format(matrix_dir))
    with open(matrix_dir / _metadata_file, 'rt') as f:
        line = f.readline()
        tokens = line.strip().split('=')
        assert tokens[0] == 'n_batches'
        n_batches = int(tokens[1])

    return [jnp.load(matrix_dir / f'batch_{batch_idx}.npy') for batch_idx in range(n_batches)]


def load_all_marginalizations(
        reads: TimeSeriesReads,
        cache: ReadStrainCollectionCache,
        read_batch_size: int,
        frag_len_negbin_n: float,
        frag_len_negbin_p: float,
        read_likelihoods: TimeSeriesLikelihoods
) -> Tuple[List[List[jnp.ndarray]], List[List[jnp.ndarray]]]:
    """
    Loads all the timepoints' marginalizations.
    Note: to save GPU memory, matrices are deleted after saved to disk. Then they are bulk-loaded from disk at the end.
    """
    # First, propagate missing matrix computations.
    for t_idx, reads_t in enumerate(reads):
        read_likelihoods_t = read_likelihoods.slices[t_idx]

        # Compute fragment frequencies.
        frag_freqs, frag_pair_freqs = FragmentFrequencyComputer(
            frag_nbinom_n=frag_len_negbin_n,
            frag_nbinom_p=frag_len_negbin_p,
            cache=cache,
            fragments=read_likelihoods_t.fragments,
            fragment_pairs=read_likelihoods_t.fragment_pairs,
            time_idx=t_idx,
            dtype=cfg.engine_cfg.dtype,
            n_threads=cfg.model_cfg.num_cores
        ).get_frequencies()

        subdir_unpaired = cache.create_subdir(Path(f'marginalizations/{t_idx}/unpaired'), recursive=True)
        if not (subdir_unpaired / _metadata_file).exists():
            compute_marginalizations_unpaired(
                reads_t=reads_t,
                read_likelihoods_t=read_likelihoods_t,
                frag_freqs=frag_freqs,
                t_idx=t_idx,
                read_batch_size=read_batch_size,
                target_dir=subdir_unpaired
            )

        subdir_paired = cache.create_subdir(Path(f'marginalizations/{t_idx}/paired'), recursive=True)
        if not (subdir_paired / _metadata_file).exists():
            compute_marginalizations_paired(
                reads_t=reads_t,
                read_likelihoods_t=read_likelihoods_t,
                frag_pair_freqs=frag_pair_freqs,
                t_idx=t_idx,
                read_batch_size=read_batch_size,
                target_dir=subdir_paired
            )

    # Next, load all the matrices from disk.
    unpaired_batches = []
    paired_batches = []
    for t_idx in range(len(reads)):
        unpaired_batches.append(
            load_batched_matrices(cache.create_subdir(Path(f'marginalizations/{t_idx}/unpaired'), recursive=True))
        )
        paired_batches.append(
            load_batched_matrices(cache.create_subdir(Path(f'marginalizations/{t_idx}/paired'), recursive=True))
        )

    # statistics
    for t_idx in range(len(reads)):
        logger.debug("Loaded {} marginalization batches for TIDX={}.".format(len(unpaired_batches[t_idx]), t_idx))
        logger.debug("Loaded {} marginalization paired batches for TIDX={}.".format(len(paired_batches[t_idx]), t_idx))
    return unpaired_batches, paired_batches


def save_batch_metadata(metadata_path: Path, n_batches: int):
    with open(metadata_path, 'wt') as f:
        print(f"n_batches={n_batches}", file=f)


def compute_marginalizations_unpaired(
        reads_t: TimeSliceReads,
        read_likelihoods_t: TimeSliceLikelihoods,
        frag_freqs: FragmentFrequencySparse,
        t_idx: int,
        read_batch_size: int,
        target_dir: Path
):
    """
    Compute unpaired reads' marginalization likelihood matrices, and store them to disk.
    """
    logger.debug(f"Computing <single-read> likelihood marginalization for t_idx={t_idx}, using batch size={read_batch_size}")
    if len(reads_t) == 0:
        logger.info("Skipping timepoint {} (t_idx={}), because there were reads".format(
            reads_t.time_point,
            t_idx
        ))
        return

    # main calculation
    logger.debug("# columns in single-read matrix: {}".format(read_likelihoods_t.lls.matrix.shape[1]))
    n_batches = 0
    for batch_idx, data_t_batch in enumerate(
            divide_columns_into_batches_sparse(
                read_likelihoods_t.lls.matrix,
                read_batch_size
            )
    ):
        logger.debug("Precomputing single-read marginalization for t = {}, batch {} ({} reads)".format(
            t_idx, batch_idx, data_t_batch.shape[1]
        ))
        # ========= Pre-compute product
        strain_batch_lls_t = log_spspmm_exp(
            frag_freqs.matrix.T,  # (S x F), note the transpose!
            data_t_batch  # F x R_batch
        )  # (S x R_batch)
        jnp.save(str(target_dir / f'batch_{batch_idx}.npy'), strain_batch_lls_t)
        n_batches += 1
    save_batch_metadata(target_dir / _metadata_file, n_batches)


def compute_marginalizations_paired(
        reads_t: TimeSliceReads,
        read_likelihoods_t: TimeSliceLikelihoods,
        frag_pair_freqs: FragmentFrequencySparse,
        t_idx: int,
        read_batch_size: int,
        target_dir: Path
):
    """
    Compute PAIRED reads' marginalization likelihood matrices, and store them to disk.
    """
    logger.debug(f"Computing <paired-read> likelihood marginalization for t_idx={t_idx}, using batch size={read_batch_size}")
    if len(reads_t) == 0:
        logger.info("Skipping timepoint {} (t_idx={}), because there were zero reads".format(
            reads_t.time_point,
            t_idx
        ))
        return []

    # main calculation
    logger.debug("# columns in paired-read matrix: {}".format(read_likelihoods_t.paired_lls.matrix.shape[1]))
    n_batches = 0
    for paired_batch_idx, paired_data_t_batch in enumerate(
        divide_columns_into_batches_sparse(
            read_likelihoods_t.paired_lls.matrix,
            read_batch_size
        )
    ):
        logger.debug(
            "Precomputing paired-read marginalization for t = {}, batch {} ({} pairs)".format(
                t_idx, paired_batch_idx, paired_data_t_batch.shape[1]
            )
        )
        # ========= Pre-compute likelihood calculations.
        batch_paired_marginalization_t = log_spspmm_exp(
            frag_pair_freqs.matrix.T,  # (S x F_pairs), note the transpose!
            paired_data_t_batch  # F_pairs x R_pairs_batch
        )  # (S x R_pairs_batch)
        jnp.save(str(target_dir / f'batch_{paired_batch_idx}.npy'), batch_paired_marginalization_t)
        n_batches += 1
    save_batch_metadata(target_dir / _metadata_file, n_batches)
