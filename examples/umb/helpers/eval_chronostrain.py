from typing import Tuple, Dict, List, Iterator
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import scipy.stats

from chronostrain.database import StrainDatabase
from chronostrain.model import Strain
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--chronostrain_dir', required=True, type=str)
    parser.add_argument('-o', '--output', required=True, type=str)
    parser.add_argument('-r', '--reads_dir', required=True, type=str)

    parser.add_argument('--group_by_clades', action='store_true')
    parser.add_argument('-c', '--clades', required=False, type=str)
    parser.add_argument('-lb', '--detection_lb', required=False, default=1e-5, type=float)
    return parser.parse_args()


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


def umb_outputs(outputs_dir: Path, read_dir: Path) -> Iterator[Tuple[str, TimeSeriesReads, np.ndarray, List[str]]]:
    for umb_dir in outputs_dir.glob("UMB*"):
        if not umb_dir.is_dir():
            raise RuntimeError(f"Expected child `{umb_dir}` to be a directory.")
        umb_id = umb_dir.name

        sample_path = umb_dir / "samples.pt"
        if not sample_path.exists():
            print(f"File `{sample_path}` not found. Skipping {umb_id}...")
            continue

        reads = TimeSeriesReads.load_from_csv(read_dir / f"{umb_id}_filtered/filtered_{umb_id}_inputs.csv")
        samples = torch.load(umb_dir / "samples.pt")
        strain_ids = load_strain_ids(umb_dir / "strains.txt")
        yield umb_id, reads, samples.cpu().numpy(), strain_ids


def overall_relabund(database_relabund: np.ndarray, reads: TimeSeriesReads, db_strains: List[Strain]) -> np.ndarray:
    """
    Converts the database-normalized relative abundances to the overall (sample-wide) relative abundance.
    """
    def total_marker_len(strain: Strain) -> int:
        ans = 0
        for marker in strain.markers:
            ans += len(marker)
        return ans

    T, N, S = database_relabund.shape
    num_filtered_reads = np.array([len(reads_t) for reads_t in reads], dtype=int)
    read_depths = np.array([reads_t.read_depth for reads_t in reads], dtype=int)
    marker_lens = np.array([total_marker_len(strain) for strain in db_strains], dtype=int)
    genome_lens = np.array([strain.metadata.total_len for strain in db_strains], dtype=int)

    marker_sum = np.sum(marker_lens.reshape((1, 1, S)) * database_relabund, axis=2)
    genome_sum = np.sum(genome_lens.reshape((1, 1, S)) * database_relabund, axis=2)
    weights = np.reshape(genome_sum / marker_sum, (T, N, 1)) * np.reshape(np.array(num_filtered_reads) / np.array(read_depths), (T, 1, 1))
    return database_relabund * weights


def load_strain_ids(strains_path: Path) -> List[str]:
    strains = []
    with open(strains_path, "rt") as f:
        for line in f:
            strains.append(line.strip())
    return strains


def evaluate(chronostrain_output_dir: Path, reads_dir: Path, db: StrainDatabase, detection_lb: float) -> pd.DataFrame:
    df_entries = []
    allowed_genera = {'Escherichia', 'Shigella'}
    for patient, reads, db_relabund_samples, _ in umb_outputs(chronostrain_output_dir, reads_dir):
        print(f"Handling {patient}.")

        thresholded_presence = np.copy(db_relabund_samples)
        thresholded_presence[thresholded_presence < detection_lb] = 0.

        filter_indices = [i for i, s in enumerate(db.all_strains()) if s.metadata.genus in allowed_genera]

        coherence = timeseries_coherence_factor(
            thresholded_presence[:, :, filter_indices]
        )

        df_entries.append({
            "Patient": patient,
            "CoherenceLower": np.quantile(coherence, q=0.025),
            "CoherenceMedian": np.quantile(coherence, q=0.5),
            "CoherenceUpper": np.quantile(coherence, q=0.975),
        })
    return pd.DataFrame(df_entries)


def evaluate_by_clades(chronostrain_output_dir: Path, reads_dir: Path, clades: Dict[str, str], db: StrainDatabase, detection_lb: float) -> pd.DataFrame:
    df_entries = []
    strains = db.all_strains()
    for patient, reads, db_relabund_samples, strain_ids in umb_outputs(chronostrain_output_dir, reads_dir):
        print(f"Handling {patient}.")
        overall_relabund_samples = overall_relabund(db_relabund_samples, reads, strains)

        for (clade, overall_chunk), (_c, relative_chunk) in zip(
                divide_into_timeseries(overall_relabund_samples, strain_ids, clades),
                divide_into_timeseries(db_relabund_samples, strain_ids, clades),
        ):
            assert clade == _c
            assert overall_chunk.shape[0] == relative_chunk.shape[0]
            assert overall_chunk.shape[1] == relative_chunk.shape[1]
            assert overall_chunk.shape[2] == relative_chunk.shape[2]

            thresholded_presence = np.copy(relative_chunk)
            thresholded_presence[thresholded_presence < detection_lb] = 0.
            coherence = timeseries_coherence_factor(relative_chunk)

            df_entries.append({
                "Patient": patient,
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


def divide_into_timeseries(timeseries: np.ndarray, strain_ids: List[str], clades: Dict[str, str]) -> Iterator[Tuple[str, np.ndarray]]:
    all_clades = sorted(list(set(clades.values())))
    for this_clade in all_clades:
        # Note: if "s" is not in "clades", then it might not be ecoli.
        matching_strain_indices = [i for i, s in enumerate(strain_ids) if (s in clades and clades[s] == this_clade)]
        if len(matching_strain_indices) == 0:
            print(f"Phylogroup {this_clade} was empty.")
            continue
        yield this_clade, timeseries[:, :, matching_strain_indices]


def timeseries_coherence_factor(x: np.ndarray) -> np.ndarray:
    return mean_coherence_factor(x[1:], x[:-1])


def mean_coherence_factor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape[0] == y.shape[0]
    assert x.shape[-1] == y.shape[-1]

    if len(x.shape) == 2 and len(y.shape) == 2:
        return np.nanmean([
            coherence_factor(x_t, y_t)
            for x_t, y_t in zip(x, y)
        ])

    if len(x.shape) == 2:
        x = np.repeat(np.expand_dims(x, 1), y.shape[1], axis=1)
    if len(y.shape) == 2:
        y = np.repeat(np.expand_dims(y, 1), x.shape[1], axis=1)

    return np.nanmean([
        [
            coherence_factor(x_tn, y_tn)
            for x_tn, y_tn in zip(x_t, y_t)
        ]
        for x_t, y_t in zip(x, y)
    ], axis=0)


def coherence_factor(x: np.ndarray, y: np.ndarray) -> float:
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


def main():
    args = parse_args()
    db = cfg.database_cfg.get_database()
    reads_dir = Path(args.reads_dir)

    detection_lb = args.detection_lb

    if args.group_by_clades:
        if args.clades is None:
            print("If grouping by clades, a clades path is required.")
            exit(1)
        clades = parse_clades(args.clades)
        df = evaluate_by_clades(Path(args.chronostrain_dir), reads_dir, clades, db, detection_lb=detection_lb)
    else:
        df = evaluate(Path(args.chronostrain_dir), reads_dir, db, detection_lb=detection_lb)

    df.to_csv(args.output, index=False, sep='\t')


if __name__ == "__main__":
    main()
