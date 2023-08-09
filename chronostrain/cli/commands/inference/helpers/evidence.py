from typing import List
from logging import Logger

from pathlib import Path
import numpy as cnp
import scipy.special
import pandas as pd
import jax
import jax.numpy as np
import jax.experimental.sparse as jsparse

from chronostrain.config import cfg
from chronostrain.algs.subroutines import SparseDataLikelihoods
from chronostrain.database import StrainDatabase
from chronostrain.model import Strain
from chronostrain.model.generative import GenerativeModel
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.cache import ComputationCache


@jax.jit
def _log_row_scale(scale, tgt_row, y_indices, y_values, answer_buf):
    """
    Given a scalar (scale), extract and scale the k-th row (tgt_row) of the matrix Y, given by a COO specification (indices, values).
    """
    v = np.where(
        y_indices[:, 0] == tgt_row,
        y_values + scale,
        -cnp.inf
    )
    return answer_buf.at[y_indices[:, 1]].max(v)


@jax.jit
def _log_col_scale(scale, tgt_col, x_indices, x_values, answer_buf):
    """
    Given a scalar (scale), extract and scale the k-th column (tgt_col) of the matrix X, given by a COO specification (indices, values).
    """
    v = np.where(
        x_indices[:, 1] == tgt_col,
        x_values + scale,
        -cnp.inf
    )
    return answer_buf.at[x_indices[:, 0]].max(v)


@jax.jit
def _spsp_tropical_mm_lax_sparsex(x_indices: np.ndarray, x_values: np.ndarray,
                                  y_indices: np.ndarray, y_values: np.ndarray,
                                  ans_buf: np.ndarray) -> np.ndarray:
    """
    jax.lax specific implementation for JIT compilation.
    Similar to log-space matrix multiplication (e.g. _log_spspmm_exp_lax_sparsex), but the inner product is now defined
        <x,y> = MAX_i (x_i + y_i)
    Or in other words, matrix multiplication in the max-tropical algebra.
    This is the X-sparse version, e.g. X @ Y = MAX_{ij} X_ij @ Y
    """
    def _helper(i, carry):
        _z, _x_indices, _x_values, _y_indices, _y_values = carry
        x_row = _x_indices[i, 0]
        x_col = _x_indices[i, 1]
        x_val = _x_values[i]
        new_row = _log_row_scale(x_val, x_col, _y_indices, _y_values, np.full(shape=_z.shape[1], fill_value=-cnp.inf))
        return (
            _z.at[x_row].set(np.maximum(_z[x_row], new_row)),
            _x_indices, _x_values, _y_indices, _y_values
        )

    return jax.lax.fori_loop(
        0, len(x_values),
        _helper,
        (ans_buf, x_indices, x_values, y_indices, y_values)
    )[0]


@jax.jit
def _spsp_tropical_mm_lax_sparsey(x_indices: np.ndarray, x_values: np.ndarray,
                                  y_indices: np.ndarray, y_values: np.ndarray,
                                  ans_buf: np.ndarray) -> np.ndarray:
    """
    Refer to _spsp_tropical_mm_lax_sparsex, but decomposes Y instead of X.
    """
    def _helper(i, carry):
        _z, _x_indices, _x_values, _y_indices, _y_values = carry
        y_row = _y_indices[i, 0]
        y_col = _y_indices[i, 1]
        y_val = _y_values[i]
        new_col = _log_col_scale(y_val, y_row, x_indices, x_values, np.full(shape=_z.shape[0], fill_value=-cnp.inf))
        return (
            _z.at[:, y_col].set(np.maximum(_z[:, y_col], new_col)),
            _x_indices, _x_values, _y_indices, _y_values
        )

    return jax.lax.fori_loop(
        0, len(y_values),
        _helper,
        (ans_buf, x_indices, x_values, y_indices, y_values)
    )[0]


def spsp_tropical_mm(x: jsparse.BCOO, y: jsparse.BCOO):
    """
    same idea as log_spspmm_exp, but assumes one is far sparser than the other.
    """
    if len(x.data) < len(y.data):
        return _spsp_tropical_mm_lax_sparsex(
            x.indices, x.data,
            y.indices, y.data,
            np.full(shape=(x.shape[0], y.shape[1]), fill_value=-cnp.inf)
        )
    else:
        return _spsp_tropical_mm_lax_sparsey(
            x.indices, x.data,
            y.indices, y.data,
            np.full(shape=(x.shape[0], y.shape[1]), fill_value=-cnp.inf)
        )


def quantify_evidence(
        db: StrainDatabase,
        model: GenerativeModel,
        data: TimeSeriesReads,
        logger: Logger,
        read_batch_size: int = 5000
):
    """Quantify evidence by calculating the best alignment score (according to the phred/random indel model)."""

    from chronostrain.algs.inference.vi.util import divide_columns_into_batches_sparse
    data_likelihoods = SparseDataLikelihoods(
        model, data, db, num_cores=cfg.model_cfg.num_cores, dtype=cfg.engine_cfg.dtype
    )

    logger.debug("Determining best per-strain, per-read likelihoods.")

    df_entries = []
    for t_idx in range(model.num_times()):
        total_sz_t = 0
        true_r_idx = -1
        for batch_idx, data_t_batch in enumerate(
                divide_columns_into_batches_sparse(
                    data_likelihoods.matrices[t_idx],
                    read_batch_size
                )
        ):
            logger.debug("Handling t = {}, batch {} ({} reads)".format(
                t_idx, batch_idx, data_t_batch.shape[1])
            )

            batch_result = spsp_tropical_mm(model.fragment_frequencies_sparse.T, data_t_batch)  # (S,R_batch) -> Raw LL
            batch_result_norm = scipy.special.softmax(batch_result, axis=0)  # (S,R_batch) -> Normalized Ratios
            batch_argsorted = np.argsort(batch_result, axis=0)
            batch_argmax = batch_argsorted[-1, :]  # highest idx
            batch_argmax2 = batch_argsorted[-2, :]  # second highest idx

            # batch_argmax = cnp.argmax(batch_result, axis=0, keepdims=False)
            # batch_norm_spread = np.std(batch_result_norm, axis=0)

            # Handle batch.
            for batch_r_idx in range(batch_argmax.shape[0]):
                true_r_idx += 1
                read = data.time_slices[t_idx][true_r_idx]
                s_idx = batch_argmax[batch_r_idx]
                s_idx2 = batch_argmax2[batch_r_idx]

                df_entries.append({
                    'T': data.time_slices[t_idx].time_point,
                    'Read': read.id,
                    'TopStrainIdx': int(batch_argmax[batch_r_idx]),
                    'LLScore': float(batch_result[s_idx, batch_r_idx]),
                    'NormScore': float(batch_result_norm[s_idx, batch_r_idx]),
                    'NormScore2': float(batch_result_norm[s_idx2, batch_r_idx])
                })
    return pd.DataFrame(
        df_entries,
        columns=['T', 'Read', 'TopStrainIdx', 'LLScore', 'NormScore', 'NormScore2']
    )



# def quantify_evidence(
#         db: StrainDatabase,
#         reads: TimeSeriesReads,
#         cache: ComputationCache,
#         strains: List[Strain],
#         t_idx: int,
#         logger: Logger
# ) -> pd.DataFrame:
#     marginalization_dir = cache.cache_dir / 'marginalizations'
#
#     if not marginalization_dir.exists():
#         logger.error("Marginalization calculations don't exist ({}). Check whether the inference ran.".format(
#             marginalization_dir
#         ))
#
#     batches_df = pd.read_csv(marginalization_dir / 'batches.tsv', sep='\t')
#     row = batches_df.loc[batches_df['T_IDX'] == t_idx, :].head(1)
#     n_batches = int(row['N_BATCHES'].item())
#     n_reads = int(row['N_READS'].item())
#
#     if n_reads != len(reads[t_idx]):
#         logger.critical("Unexpected error: batches.tsv reported {} reads, but input parsed {} reads.".format(
#             n_reads,
#             len(reads[t_idx])
#         ))
#         exit(1)
#
#     strain_subset = {s.id for s in strains}
#     strain_indices = [
#         s_idx
#         for s_idx, s in enumerate(db.all_strains())
#         if s.id in strain_subset
#     ]
#
#     running_max = np.full(len(strains), dtype=float, fill_value=-np.inf)
#     running_argmax = np.full(len(strains), dtype=int, fill_value=-1)
#     running_threshold_count_95 = np.zeros(len(strains), dtype=int)
#     running_threshold_count_75 = np.zeros(len(strains), dtype=int)
#     running_threshold_count_50 = np.zeros(len(strains), dtype=int)
#     running_threshold_count_25 = np.zeros(len(strains), dtype=int)
#     running_threshold_count_05 = np.zeros(len(strains), dtype=int)
#
#     batch_offset = 0
#     for batch_idx in range(n_batches):
#         f = marginalization_dir / f"t_{t_idx}_batch_{batch_idx}.npy"
#         batch_lls = np.load(str(f))  # S x F_batch
#         batch_lls = batch_lls[strain_indices, :]
#
#         batch_p = batch_lls
#
#         # # Normalize across strains (to estimate posterior contribution)
#         # batch_p = scipy.special.softmax(batch_lls, axis=0)
#         # del batch_lls
#
#         batch_max = np.nanmax(batch_p, axis=1, initial=-np.inf, keepdims=False)
#         batch_argmax = batch_offset + np.nanargmax(batch_p, axis=1, keepdims=False)
#
#         running_argmax = np.where(running_max > batch_max, running_argmax, batch_argmax)
#         running_max = np.where(running_max > batch_max, running_max, batch_max)
#
#         running_threshold_count_95 = np.sum(
#             np.stack([
#                 running_threshold_count_95,
#                 np.greater(batch_p, np.log(0.95)).sum(axis=1)
#             ], axis=0),
#             axis=0
#         )
#
#         running_threshold_count_75 = np.sum(
#             np.stack([
#                 running_threshold_count_75,
#                 np.greater(batch_p, np.log(0.75)).sum(axis=1)
#             ], axis=0),
#             axis=0
#         )
#
#         running_threshold_count_50 = np.sum(
#             np.stack([
#                 running_threshold_count_50,
#                 np.greater(batch_p, np.log(0.50)).sum(axis=1)
#             ], axis=0),
#             axis=0
#         )
#
#         running_threshold_count_25 = np.sum(
#             np.stack([
#                 running_threshold_count_25,
#                 np.greater(batch_p, np.log(0.25)).sum(axis=1)
#             ], axis=0),
#             axis=0
#         )
#
#         running_threshold_count_05 = np.sum(
#             np.stack([
#                 running_threshold_count_05,
#                 np.greater(batch_p, np.log(0.05)).sum(axis=1)
#             ], axis=0),
#             axis=0
#         )
#
#         batch_offset += batch_p.shape[1]
#
#     if batch_offset != n_reads:
#         logger.critical("Total shape ({}) does not match the input read count ({}) for t_idx = {}!".format(
#             batch_offset,
#             n_reads,
#             t_idx
#         ))
#         exit(1)
#
#     # Create Dataframe.
#     df_entries = []
#     for strain_idx, strain in enumerate(strains):
#         if running_argmax[strain_idx] >= 0:
#             best_read = reads[t_idx][running_argmax[strain_idx]].id
#         else:
#             best_read = ""
#         df_entries.append({
#             "Strain": strain.id,
#             "MaxEvidence": running_max[strain_idx],
#             "MaxEvidenceRead": best_read,
#             "CountThreshold95": running_threshold_count_95[strain_idx],
#             "CountThreshold75": running_threshold_count_75[strain_idx],
#             "CountThreshold50": running_threshold_count_50[strain_idx],
#             "CountThreshold25": running_threshold_count_25[strain_idx],
#             "CountThreshold05": running_threshold_count_05[strain_idx],
#             "NumFiltReads": len(reads[t_idx]),
#             "ReadDepth": reads[t_idx].read_depth
#         })
#     return pd.DataFrame(df_entries)
