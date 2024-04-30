import itertools
from typing import Tuple, Dict, List, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from chronostrain.model import Strain
from .chronostrain_result import ChronostrainResult, Taxon


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
        abund_lb: float
) -> pd.DataFrame:
    df_entries = []
    for umb_result in chronostrain_outputs:
        print(f"Computing correlations for {umb_result.name}")
        df = umb_result.annot_df_with_lower_bound(abund_lb, target_taxon=Taxon("Escherichia", "coli"))
        df['Phylogroup'] = df['StrainId'].map(clades)

        for clade, clade_section in df.groupby("Phylogroup"):
            coherence, group_sz, abund = timeseries_coherence(umb_result.time_points, clade_section, umb_result)
            df_entries.append({
                "Patient": umb_result.name,
                "Phylogroup": clade,
                "GroupSize": group_sz,
                "Coherence": coherence,
                "Abundance": abund
            })
    return pd.DataFrame(df_entries)


def timeseries_coherence(
        time_points: List[float],
        clade_section: pd.DataFrame,
        res: ChronostrainResult
) -> Tuple[float, int, float]:
    strain_idxs = sorted(pd.unique(clade_section['StrainIdx']))
    mat = np.zeros((len(time_points), len(strain_idxs)), dtype=float)
    overall_abund = res.overall_ra()
    for t_idx, t in enumerate(time_points):
        for mat_idx, s_idx in enumerate(strain_idxs):
            mat[t_idx, mat_idx] = np.median(overall_abund[t_idx, :, s_idx])

    avg_corr = timeseries_coherence_factor(mat)
    return avg_corr, len(strain_idxs), np.max(mat.sum(axis=-1))


# def divide_into_timeseries(timeseries: np.ndarray, strains: List[Strain], clades: Dict[str, str]) -> Iterator[Tuple[str, np.ndarray]]:
#     all_clades = sorted(list(set(clades.values())))
#     for this_clade in all_clades:
#         # Note: if "s" is not in "clades", then it might not be ecoli.
#         matching_strain_indices = [i for i, s in enumerate(strains) if (s.id in clades and clades[s.id] == this_clade)]
#         if len(matching_strain_indices) == 0:
#             print(f"Phylogroup {this_clade} was empty.")
#             continue
#         yield this_clade, timeseries[:, :, matching_strain_indices]
#
#
def timeseries_coherence_factor(x: np.ndarray) -> float:
    vars = np.var(x, axis=-1)
    if np.sum(vars > 0) == 0:
        return np.nan
    x = x[np.where(vars > 0)[0], :]
    return mean_correlation_factor(x[1:], x[:-1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: 1-d array
    :param y: 1-d array
    :return:
    """
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return scipy.stats.spearmanr(x, y).correlation


def mean_correlation_factor(x: np.ndarray, y: np.ndarray) -> float:
    return np.nanmean([
        spearman_corr(x_t, y_t)
        for x_t, y_t in zip(x, y)
    ], axis=0)


def analyze_correlations(chronostrain_outputs: List[ChronostrainResult],
                         clades_tsv: Path,
                         abund_lb: float):
    clades = parse_clades(clades_tsv)
    df = evaluate_by_clades(chronostrain_outputs, clades, abund_lb)
    return df
