from typing import Tuple, Dict, List, Iterator
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--chronostrain_dir', required=True, type=str)
    parser.add_argument('-o', '--output', required=True, type=str)

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


def umb_outputs(base_dir: Path) -> Iterator[Tuple[str, torch.Tensor, List[str]]]:
    for umb_dir in base_dir.glob("UMB*"):
        if not umb_dir.is_dir():
            raise RuntimeError(f"Expected child `{umb_dir}` to be a directory.")
        umb_id = umb_dir.name

        sample_path = umb_dir / "samples.pt"
        if not sample_path.exists():
            print(f"File `{sample_path}` not found. Skipping {umb_id}...")
            continue

        samples = torch.load(umb_dir / "samples.pt")
        strain_ids = load_strain_ids(umb_dir / "strains.txt")
        yield umb_id, samples, strain_ids


def load_strain_ids(strains_path: Path) -> List[str]:
    strains = []
    with open(strains_path, "rt") as f:
        for line in f:
            strains.append(line.strip())
    return strains


def evaluate(chronostrain_output_dir: Path) -> pd.DataFrame:
    df_entries = []
    for patient, umb_samples, _ in umb_outputs(chronostrain_output_dir):
        print(f"Handling {patient}.")
        timeseries = torch.median(umb_samples, dim=1).values.numpy()

        df_entries.append({
            "Patient": patient,
            "Dominance": dominance_switch_ratio(timeseries, lb=1 / timeseries.shape[1])
        })
    return pd.DataFrame(df_entries)


def evaluate_by_clades(chronostrain_output_dir: Path, clades: Dict[str, str]) -> pd.DataFrame:
    df_entries = []
    for patient, umb_samples, strain_ids in umb_outputs(chronostrain_output_dir):
        print(f"Handling {patient}.")
        timeseries = torch.median(umb_samples, dim=1).values.numpy()

        for clade, sub_timeseries in divide_into_timeseries(timeseries, strain_ids, clades):
            df_entries.append({
                "Patient": patient,
                "Phylogroup": clade,
                "GroupSize": sub_timeseries.shape[1],
                "Dominance": dominance_switch_ratio(sub_timeseries, lb=1 / timeseries.shape[1]),
                "OverallRelAbundMax": np.max(np.sum(sub_timeseries, axis=1)),
                "StrainRelAbundMax": np.max(sub_timeseries)
            })
    return pd.DataFrame(df_entries)


def divide_into_timeseries(timeseries: np.ndarray, strain_ids: List[str], clades: Dict[str, str]) -> Iterator[Tuple[str, np.ndarray]]:
    all_clades = set(clades.values())
    for this_clade in all_clades:
        # Note: if "s" is not in "clades", then it might not be ecoli.
        matching_strains = [i for i, s in enumerate(strain_ids) if (s in clades and clades[s] == this_clade)]
        if len(matching_strains) == 0:
            print(f"Phylogroup {this_clade} was empty.")
            continue
        yield this_clade, timeseries[:, matching_strains]


def dominance_switch_ratio(abundance_est: np.ndarray, lb: float = 0.0) -> float:
    """
    Calculate how often the dominant strain switches.
    """
    dom = np.argmax(abundance_est, axis=1)
    num_switches = 0

    def row_is_zeros(r) -> bool:
        return np.sum(r <= lb).item() == r.shape[0]

    num_total = 0
    for i in range(len(dom) - 1):
        if row_is_zeros(abundance_est[i]) and row_is_zeros(abundance_est[i + 1]):
            continue  # don't add to denominator.

        elif (dom[i] != dom[i + 1]) or row_is_zeros(abundance_est[i]) or row_is_zeros(abundance_est[i + 1]):
            num_switches += 1
        num_total += 1
    if num_total > 0:
        return num_switches / num_total
    else:
        return np.nan


def main():
    args = parse_args()

    if args.group_by_clades:
        if args.clades is None:
            print("If grouping by clades, a clades path is required.")
            exit(1)
        clades = parse_clades(args.clades)
        df = evaluate_by_clades(Path(args.chronostrain_dir), clades)
    else:
        df = evaluate(Path(args.chronostrain_dir))

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
