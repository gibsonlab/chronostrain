from typing import List
from logging import Logger

import numpy as np
import scipy.special
import pandas as pd

from chronostrain.database import StrainDatabase
from chronostrain.model import Strain
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.cache import ComputationCache


def quantify_evidence(
        db: StrainDatabase,
        reads: TimeSeriesReads,
        cache: ComputationCache,
        strains: List[Strain],
        t_idx: int,
        logger: Logger
) -> pd.DataFrame:
    marginalization_dir = cache.cache_dir / 'marginalizations'

    if not marginalization_dir.exists():
        logger.error("Marginalization calculations don't exist ({}). Check whether the inference ran.".format(
            marginalization_dir
        ))

    batches_df = pd.read_csv(marginalization_dir / 'batches.tsv', sep='\t')
    row = batches_df.loc[batches_df['T_IDX'] == t_idx, :].head(1)
    n_batches = int(row['N_BATCHES'].item())
    n_reads = int(row['N_READS'].item())

    if n_reads != len(reads[t_idx]):
        logger.critical("Unexpected error: batches.tsv reported {} reads, but input parsed {} reads.".format(
            n_reads,
            len(reads[t_idx])
        ))
        exit(1)

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

    if batch_offset != n_reads:
        logger.critical("Total shape ({}) does not match the input read count ({}) for t_idx = {}!".format(
            batch_offset,
            n_reads,
            t_idx
        ))
        exit(1)

    # Create Dataframe.
    df_entries = []
    for strain_idx, strain in enumerate(strains):
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
