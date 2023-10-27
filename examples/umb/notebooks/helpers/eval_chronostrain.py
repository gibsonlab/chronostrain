from typing import Tuple, Dict, List, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from chronostrain.model import Strain
from chronostrain.config import cfg
from .chronostrain_result import ChronostrainResult


def strip_suffixes(x):
    x = Path(x)
    suffix_set = {'.chrom', 'fa', '.fna', '.gz', 'fasta'}
    while x.suffix in suffix_set:
        x = x.with_suffix('')
    return x.name


def parse_clades(clades_path: Path) -> Dict[str, str]:
    """
    NC_017626.1.chrom.fna	['ybgD', 'trpA', 'trpBA', 'chuA', 'arpA', 'trpAgpC']	['+', '+', '-', '-']	['trpAgpC']	D	NC_017626.1.chrom.fna_mash_screen.tab
    """
    mapping = {}
    with open(clades_path, "rt") as clades_file:
        for line in clades_file:
            line = line.strip()
            if len(line) == 0:
                continue

            tokens = line.split('\t')
            strain_id = strip_suffixes(tokens[0])
            phylogroup = tokens[4]
            mapping[strain_id] = phylogroup
    return mapping


def evaluate_by_clades(
        chronostrain_outputs: List[ChronostrainResult],
        clades: Dict[str, str],
) -> pd.DataFrame:
    df_entries = []
    for umb_result in chronostrain_outputs:
        overall_relabund_samples = umb_result.overall_ra()
        db_relabund_samples = umb_result.filt_ra()
        strains = umb_result.display_strains

        for (clade, overall_chunk), (_c, relative_chunk) in zip(
                divide_into_timeseries(overall_relabund_samples, strains, clades),
                divide_into_timeseries(db_relabund_samples, strains, clades),
        ):
            assert clade == _c
            assert overall_chunk.shape[0] == relative_chunk.shape[0]
            assert overall_chunk.shape[1] == relative_chunk.shape[1]
            assert overall_chunk.shape[2] == relative_chunk.shape[2]

            coherence = timeseries_coherence_factor(relative_chunk)
            df_entries.append({
                "Patient": umb_result.name,
                "Phylogroup": clade,
                "GroupSize": overall_chunk.shape[-1],
                "CoherenceLower": np.quantile(coherence, q=0.025),
                "CoherenceMedian": np.quantile(coherence, q=0.5),
                "CoherenceUpper": np.quantile(coherence, q=0.975),
                "CladeOverallRelAbundLower": np.max(np.quantile(overall_chunk.sum(axis=-1), q=0.025, axis=1)),
                "CladeOverallRelAbundMedian": np.max(np.quantile(overall_chunk.sum(axis=-1), q=0.5, axis=1)),
                "CladeOverallRelAbundUpper": np.max(np.quantile(overall_chunk.sum(axis=-1), q=0.975, axis=1)),
                "StrainRelAbundLower": np.max(np.quantile(relative_chunk, axis=1, q=0.025)),
                "StrainRelAbundMedian": np.max(np.quantile(relative_chunk, axis=1, q=0.5)),
                "StrainRelAbundUppser": np.max(np.quantile(relative_chunk, axis=1, q=0.975))
            })
    return pd.DataFrame(df_entries)


def divide_into_timeseries(timeseries: np.ndarray, strains: List[Strain], clades: Dict[str, str]) -> Iterator[Tuple[str, np.ndarray]]:
    all_clades = sorted(list(set(clades.values())))
    for this_clade in all_clades:
        # Note: if "s" is not in "clades", then it might not be ecoli.
        matching_strain_indices = [i for i, s in enumerate(strains) if (s.id in clades and clades[s.id] == this_clade)]
        if len(matching_strain_indices) == 0:
            print(f"Phylogroup {this_clade} was empty.")
            continue
        yield this_clade, timeseries[:, :, matching_strain_indices]


def timeseries_coherence_factor(x: np.ndarray) -> np.ndarray:
    return mean_correlation_factor(x[1:], x[:-1])


def mean_correlation_factor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape[0] == y.shape[0]
    assert x.shape[-1] == y.shape[-1]

    if len(x.shape) == 2 and len(y.shape) == 2:
        return np.nanmean([
            correlation_factor(x_t, y_t)
            for x_t, y_t in zip(x, y)
        ])

    if len(x.shape) == 2:
        x = np.repeat(np.expand_dims(x, 1), y.shape[1], axis=1)
    if len(y.shape) == 2:
        y = np.repeat(np.expand_dims(y, 1), x.shape[1], axis=1)

    return np.nanmean([
        [
            correlation_factor(x_tn, y_tn)
            for x_tn, y_tn in zip(x_t, y_t)
        ]
        for x_t, y_t in zip(x, y)
    ], axis=0)


def correlation_factor(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x.shape) == 1
    assert len(y.shape) == 1

    if np.std(x) == 0 and np.std(y) == 0:  # edge case
        if x[0] == 0.:
            return np.nan
        elif x[0] == y[0]:
            return 1.0
        else:
            return 0.0

    if np.std(x) == 0 or np.std(y) == 0:  # only one is zero
        return 0.0

    # noinspection PyTypeChecker
    return scipy.stats.spearmanr(x, y)[0]


def analyze_correlations(chronostrain_outputs: List[ChronostrainResult], clades_tsv: Path, output_path: Path):
    clades = parse_clades(clades_tsv)
    df = evaluate_by_clades(chronostrain_outputs, clades)
    df.to_csv(output_path, index=False, sep='\t')
