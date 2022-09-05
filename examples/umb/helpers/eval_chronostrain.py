from typing import Tuple, Dict, List, Iterator
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

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


def evaluate(chronostrain_output_dir: Path, reads_dir: Path) -> pd.DataFrame:
    df_entries = []
    for patient, reads, db_relabund_samples, _ in umb_outputs(chronostrain_output_dir, reads_dir):
        print(f"Handling {patient}.")

        df_entries.append({
            "Patient": patient,
            "Dominance": dominance_switch_ratio(db_relabund_samples)
        })
    return pd.DataFrame(df_entries)


def evaluate_by_clades(chronostrain_output_dir: Path, reads_dir: Path, clades: Dict[str, str], db: StrainDatabase) -> pd.DataFrame:
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

            df_entries.append({
                "Patient": patient,
                "Phylogroup": clade,
                "GroupSize": overall_chunk.shape[1],
                "DominanceLower": np.quantile(dominance_switch_ratio(relative_chunk), q=0.025),
                "DominanceMedian": np.quantile(dominance_switch_ratio(relative_chunk), q=0.5),
                "DominanceUpper": np.quantile(dominance_switch_ratio(relative_chunk), q=0.975),
                "CladeOverallRelAbundLower": np.quantile(overall_chunk.sum(axis=-1).max(axis=0), q=0.025),
                "CladeOverallRelAbundMedian": np.quantile(overall_chunk.sum(axis=-1).max(axis=0), q=0.5),
                "CladeOverallRelAbundUpper": np.quantile(overall_chunk.sum(axis=-1).max(axis=0), q=0.975),
                "StrainRelAbundLower": np.quantile(relative_chunk.max(axis=-1, initial=0.).max(axis=0), q=0.025),
                "StrainRelAbundMedian": np.quantile(relative_chunk.max(axis=-1, initial=0.).max(axis=0), q=0.5),
                "StrainRelAbundUppser": np.quantile(relative_chunk.max(axis=-1, initial=0.).max(axis=0), q=0.975)
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
        yield this_clade, timeseries[:, matching_strain_indices]


def dominance_switch_ratio(abundance_est: np.ndarray, lb: float = 0.0) -> np.ndarray:
    """
    Calculate how often the dominant strain switches.
    """
    def clade_is_missing(r) -> np.ndarray:
        return np.sum(r > lb, axis=-1) == 0

    if len(abundance_est.shape) == 3:
        _T, _N, _S = abundance_est.shape
        _is_missing = np.zeros((_T - 1, _N), dtype=np.bool)
        _has_switch = np.zeros((_T - 1, _N), dtype=np.bool)
    else:
        _T, _S = abundance_est.shape
        _is_missing = np.zeros(_T, dtype=np.bool)
        _has_switch = np.zeros(_T, dtype=np.bool)

    dom = np.argmax(abundance_est, axis=-1)
    for t in range(_T - 1):
        _is_missing[t] = clade_is_missing(abundance_est[t])
        _has_switch[t] = abundance_est[t + 1] or dom[t] != dom[t + 1]

    # Compute conditional ratio.
    numer = np.sum(np.logical_not(_is_missing) and _has_switch, axis=0)
    denom = np.sum(np.logical_not(_is_missing), axis=0)
    return numer / denom


def main():
    args = parse_args()
    db = cfg.database_cfg.get_database()
    reads_dir = Path(args.reads_dir)

    if args.group_by_clades:
        if args.clades is None:
            print("If grouping by clades, a clades path is required.")
            exit(1)
        clades = parse_clades(args.clades)
        df = evaluate_by_clades(Path(args.chronostrain_dir), reads_dir, clades, db)
    else:
        df = evaluate(Path(args.chronostrain_dir), reads_dir)

    df.to_csv(args.output, index=False, sep='\t')


if __name__ == "__main__":
    main()
