# from typing import Set
# from logging import Logger
#
# import numpy as cnp
# import scipy.special
# import pandas as pd
# import jax
# import jax.numpy as np
# import jax.experimental.sparse as jsparse
#
# from chronostrain.config import cfg
# from chronostrain.database import StrainDatabase
# from chronostrain.model.generative import AbundanceGaussianPrior
# from chronostrain.model.io import TimeSeriesReads
#
#
# @jax.jit
# def _log_row_scale(scale, tgt_row, y_indices, y_values, answer_buf):
#     """
#     Given a scalar (scale), extract and scale the k-th row (tgt_row) of the matrix Y, given by a COO specification (indices, values).
#     """
#     v = np.where(
#         y_indices[:, 0] == tgt_row,
#         y_values + scale,
#         -cnp.inf
#     )
#     return answer_buf.at[y_indices[:, 1]].max(v)
#
#
# @jax.jit
# def _log_col_scale(scale, tgt_col, x_indices, x_values, answer_buf):
#     """
#     Given a scalar (scale), extract and scale the k-th column (tgt_col) of the matrix X, given by a COO specification (indices, values).
#     """
#     v = np.where(
#         x_indices[:, 1] == tgt_col,
#         x_values + scale,
#         -cnp.inf
#     )
#     return answer_buf.at[x_indices[:, 0]].max(v)
#
#
# @jax.jit
# def _spsp_tropical_mm_lax_sparsex(x_indices: np.ndarray, x_values: np.ndarray,
#                                   y_indices: np.ndarray, y_values: np.ndarray,
#                                   ans_buf: np.ndarray) -> np.ndarray:
#     """
#     jax.lax specific implementation for JIT compilation.
#     Similar to log-space matrix multiplication (e.g. _log_spspmm_exp_lax_sparsex), but the inner product is now defined
#         <x,y> = MAX_i (x_i + y_i)
#     Or in other words, matrix multiplication in the max-tropical algebra.
#     This is the X-sparse version, e.g. X @ Y = MAX_{ij} X_ij @ Y
#     """
#     def _helper(i, carry):
#         _z, _x_indices, _x_values, _y_indices, _y_values = carry
#         x_row = _x_indices[i, 0]
#         x_col = _x_indices[i, 1]
#         x_val = _x_values[i]
#         new_row = _log_row_scale(x_val, x_col, _y_indices, _y_values, np.full(shape=_z.shape[1], fill_value=-cnp.inf))
#         return (
#             _z.at[x_row].set(np.maximum(_z[x_row], new_row)),
#             _x_indices, _x_values, _y_indices, _y_values
#         )
#
#     return jax.lax.fori_loop(
#         0, len(x_values),
#         _helper,
#         (ans_buf, x_indices, x_values, y_indices, y_values)
#     )[0]
#
#
# @jax.jit
# def _spsp_tropical_mm_lax_sparsey(x_indices: np.ndarray, x_values: np.ndarray,
#                                   y_indices: np.ndarray, y_values: np.ndarray,
#                                   ans_buf: np.ndarray) -> np.ndarray:
#     """
#     Refer to _spsp_tropical_mm_lax_sparsex, but decomposes Y instead of X.
#     """
#     def _helper(i, carry):
#         _z, _x_indices, _x_values, _y_indices, _y_values = carry
#         y_row = _y_indices[i, 0]
#         y_col = _y_indices[i, 1]
#         y_val = _y_values[i]
#         new_col = _log_col_scale(y_val, y_row, x_indices, x_values, np.full(shape=_z.shape[0], fill_value=-cnp.inf))
#         return (
#             _z.at[:, y_col].set(np.maximum(_z[:, y_col], new_col)),
#             _x_indices, _x_values, _y_indices, _y_values
#         )
#
#     return jax.lax.fori_loop(
#         0, len(y_values),
#         _helper,
#         (ans_buf, x_indices, x_values, y_indices, y_values)
#     )[0]
#
#
# def spsp_tropical_mm(x: jsparse.BCOO, y: jsparse.BCOO):
#     """
#     same idea as log_spspmm_exp, but assumes one is far sparser than the other.
#     """
#     if len(x.data) < len(y.data):
#         return _spsp_tropical_mm_lax_sparsex(
#             x.indices, x.data,
#             y.indices, y.data,
#             np.full(shape=(x.shape[0], y.shape[1]), fill_value=-cnp.inf)
#         )
#     else:
#         return _spsp_tropical_mm_lax_sparsey(
#             x.indices, x.data,
#             y.indices, y.data,
#             np.full(shape=(x.shape[0], y.shape[1]), fill_value=-cnp.inf)
#         )
#
#
# def quantify_evidence(
#         db: StrainDatabase,
#         model: AbundanceGaussianPrior,
#         data: TimeSeriesReads,
#         logger: Logger,
#         target_strain_ids: Set[str],
#         read_batch_size: int = 5000
# ):
#     """Quantify evidence by calculating the best alignment score (according to the phred/random indel model)."""
#     target_strain_idx = [
#         i
#         for i, s in enumerate(db.all_strains())
#         if s.id in target_strain_ids
#     ]
#
#     from chronostrain.inference.algs.vi.base.util import divide_columns_into_batches_sparse
#     data_likelihoods = SparseDataLikelihoods(
#         model, data, db, num_cores=cfg.model_cfg.num_cores, dtype=cfg.engine_cfg.dtype
#     )
#
#     logger.debug("Determining best per-strain, per-read likelihoods.")
#
#     df_entries = []
#     for t_idx in range(model.num_times()):
#         true_r_idx = -1
#         for batch_idx, data_t_batch in enumerate(
#                 divide_columns_into_batches_sparse(
#                     data_likelihoods.matrices[t_idx],
#                     read_batch_size
#                 )
#         ):
#             logger.debug("Handling t = {}, batch {} ({} reads)".format(
#                 t_idx, batch_idx, data_t_batch.shape[1])
#             )
#
#             batch_result = spsp_tropical_mm(model.fragment_frequencies_sparse.T, data_t_batch)  # (S,R_batch) -> Raw LL
#             batch_result = batch_result[target_strain_idx, :]
#
#             batch_result_norm = scipy.special.softmax(batch_result, axis=0)  # (S,R_batch) -> Normalized Ratios
#             batch_argsorted = np.argsort(batch_result, axis=0)
#             batch_argmax = batch_argsorted[-1, :]  # highest idx
#             batch_argmax2 = batch_argsorted[-2, :]  # second highest idx
#
#             # batch_argmax = cnp.argmax(batch_result, axis=0, keepdims=False)
#             # batch_norm_spread = np.std(batch_result_norm, axis=0)
#
#             # Handle batch.
#             for batch_r_idx in range(batch_argmax.shape[0]):
#                 true_r_idx += 1
#                 read = data.time_slices[t_idx][true_r_idx]
#                 s_idx = batch_argmax[batch_r_idx]
#                 s_idx2 = batch_argmax2[batch_r_idx]
#
#                 df_entries.append({
#                     'T': data.time_slices[t_idx].time_point,
#                     'Read': read.id,
#                     'TopStrainIdx': int(batch_argmax[batch_r_idx]),
#                     'LLScore': float(batch_result[s_idx, batch_r_idx]),
#                     'NormScore': float(batch_result_norm[s_idx, batch_r_idx]),
#                     'NormScore2': float(batch_result_norm[s_idx2, batch_r_idx]),
#                     'TopStrainIdx2': int(batch_argmax2[batch_r_idx]),
#                 })
#     return pd.DataFrame(
#         df_entries,
#         columns=['T', 'Read', 'TopStrainIdx', 'LLScore', 'NormScore', 'NormScore2', 'TopStrainIdx2']
#     )