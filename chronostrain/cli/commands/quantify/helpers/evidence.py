from typing import List
from logging import Logger

import numpy as np
import scipy.special
import pandas as pd

from chronostrain.algs.subroutines.cache import ReadsPopulationCache
from chronostrain.database import StrainDatabase
from chronostrain.model import Strain
from chronostrain.model.io import TimeSeriesReads


def quantify_evidence(
        db: StrainDatabase,
        reads: TimeSeriesReads,
        strains: List[Strain],
        t_idx: int,
        logger: Logger
) -> pd.DataFrame:
    cache = ReadsPopulationCache(reads, db)
    marginalization_dir = cache.cache_dir / 'marginalizations'

    if not marginalization_dir.exists():
        frag_ll_dir = cache.cache_dir / 'fragment_frequencies'
        read_batch_size = 10000

        if frag_ll_dir.exists():  # =========== BEGIN temporary cache fix code here
            logger.info("Outdated cache found. Temporary solution: recalculating marginalizations.")
            TMP_dir = cache.cache_dir / 'marginalizations_TMP'
            if TMP_dir.exists():
                for c in TMP_dir.iterdir():
                    c.unlink()
                TMP_dir.rmdir()
            TMP_dir.mkdir(exist_ok=True)

            # re-do calculations from ADVI initializer.  --> TODO phase this code out.
            from chronostrain.algs.inference.vi.util import divide_columns_into_batches_sparse
            from chronostrain.util.math import load_sparse_matrix
            from chronostrain.config import cfg
            from chronostrain.util.math import log_spspmm_exp

            fragment_frequencies_sparse = load_sparse_matrix(
                cache.cache_dir / 'fragment_frequencies' / f'sparse_frag_freqs.{cfg.engine_cfg.dtype}.npz'
            )

            def _compute_marginalization(_t_idx, _batch_idx, _data_t_batch):
                logger.debug("Precomputing marginalization for t = {}, batch {} ({} reads)".format(
                    _t_idx, _batch_idx, _data_t_batch.shape[1]
                ))
                # ========= Pre-compute likelihood calculations.
                return log_spspmm_exp(
                    fragment_frequencies_sparse.T,  # (S x F), note the transpose!
                    _data_t_batch  # F x R_batch
                )  # (S x R_batch)

            for t_idx in range(len(reads)):
                read_ll_t = load_sparse_matrix(cache.cache_dir / 'log_likelihoods' / f'sparse_log_likelihoods_{t_idx}.{cfg.engine_cfg.dtype}.npz')
                for batch_idx, data_t_batch in enumerate(
                    divide_columns_into_batches_sparse(
                        read_ll_t,
                        read_batch_size
                    )
                ):
                    _ = cache.call(
                        relative_filepath=f'marginalizations_TMP/t_{t_idx}_batch_{batch_idx}.npy',
                        fn=_compute_marginalization,
                        call_args=[t_idx, batch_idx, data_t_batch],
                        save=lambda p, x: np.save(p, x),
                        load=lambda p: np.load(p),
                    )
            TMP_dir.rename(marginalization_dir)
        else:  # =========== END temporary cache fix code here
            logger.error("marginalizations exist but fragment_frequencies does not. Check whether the inference ran.")


    strain_subset = {s.id for s in strains}
    strain_indices = [
        s_idx
        for s_idx, s in enumerate(db.all_strains())
        if s.id in strain_subset
    ]

    running_max = np.zeros(len(strains), dtype=float)
    running_argmax = np.full(len(strains), dtype=int, fill_value=-1)
    running_threshold_count_95 = np.zeros(len(strains), dtype=int)
    running_threshold_count_75 = np.zeros(len(strains), dtype=int)
    running_threshold_count_50 = np.zeros(len(strains), dtype=int)
    running_threshold_count_25 = np.zeros(len(strains), dtype=int)
    running_threshold_count_05 = np.zeros(len(strains), dtype=int)

    n_batches = len(list(marginalization_dir.glob(f"t_{t_idx}_batch_*.npy")))
    batch_offset = 0
    for batch_idx in range(n_batches):
        f = marginalization_dir / f"t_{t_idx}_batch_{batch_idx}.npy"
        batch_lls = np.load(str(f))  # S x F_batch
        batch_lls = batch_lls[strain_indices, :]

        # Normalize across strains (to estimate posterior contribution)
        batch_p = scipy.special.softmax(batch_lls, axis=0)
        del batch_lls

        batch_max = np.nanmax(batch_p, axis=1, initial=0., keepdims=False)
        batch_argmax = batch_offset + np.nanargmax(batch_p, axis=1, keepdims=False)

        running_argmax = np.where(running_max > batch_max, running_argmax, batch_argmax)
        running_max = np.where(running_max > batch_max, running_max, batch_max)

        running_threshold_count_95 = np.sum(
            np.stack([
                running_threshold_count_95,
                np.greater(batch_p, 0.95).sum(axis=1)
            ], axis=0),
            axis=0
        )

        running_threshold_count_75 = np.sum(
            np.stack([
                running_threshold_count_75,
                np.greater(batch_p, 0.75).sum(axis=1)
            ], axis=0),
            axis=0
        )

        running_threshold_count_50 = np.sum(
            np.stack([
                running_threshold_count_50,
                np.greater(batch_p, 0.50).sum(axis=1)
            ], axis=0),
            axis=0
        )

        running_threshold_count_25 = np.sum(
            np.stack([
                running_threshold_count_25,
                np.greater(batch_p, 0.25).sum(axis=1)
            ], axis=0),
            axis=0
        )

        running_threshold_count_05 = np.sum(
            np.stack([
                running_threshold_count_05,
                np.greater(batch_p, 0.05).sum(axis=1)
            ], axis=0),
            axis=0
        )

        batch_offset += batch_p.shape[1]

    if batch_offset != len(reads[t_idx]):
        logger.critical("Total shape ({}) does not match the input read count ({}) for timepoint t = {}!".format(
            batch_offset,
            len(reads[t_idx]),
            reads[t_idx].time_point
        ))
        exit(1)

    # Create Dataframe.
    df_entries = []
    for strain_idx, strain in enumerate(strains):
        # print(running_argmax[strain_idx])
        # print(reads[t_idx].reads)
        if running_argmax[strain_idx] >= 0:
            best_read = reads[t_idx][running_argmax[strain_idx]].id
        else:
            best_read = ""
        df_entries.append({
            "Strain": strain.id,
            "MaxEvidence": running_max[strain_idx],
            "MaxEvidenceRead": best_read,
            "CountThreshold95": running_threshold_count_95[strain_idx],
            "CountThreshold75": running_threshold_count_75[strain_idx],
            "CountThreshold50": running_threshold_count_50[strain_idx],
            "CountThreshold25": running_threshold_count_25[strain_idx],
            "CountThreshold05": running_threshold_count_05[strain_idx],
            "NumFiltReads": len(reads[t_idx]),
            "ReadDepth": reads[t_idx].read_depth
        })
    return pd.DataFrame(df_entries)
