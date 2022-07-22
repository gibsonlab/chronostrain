import argparse
from pathlib import Path
from typing import Tuple, Iterator

import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sb

from chronostrain import cfg
from chronostrain.database import StrainDatabase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Take all chronostrain runs and evaluate their performances.")

    parser.add_argument('-b', '--base_data_dir', required=True, type=str,
                        help='<Required> The data directory contining all of the runs.')
    parser.add_argument('-g', '--ground_truth_path', required=True, type=str,
                        help='<Required> The path to the ground truth file.')
    parser.add_argument('-o', '--out_path', required=True, type=str,
                        help='<Required> The output path to save the data to. (Output is CSV format)')

    return parser.parse_args()


def read_depth_dirs(base_dir: Path) -> Iterator[Tuple[int, Path]]:
    for child_dir in base_dir.glob("reads_*"):
        if not child_dir.is_dir():
            raise RuntimeError(f"Expected child `{child_dir}` to be a directory.")

        read_depth = int(child_dir.name.split("_")[1])
        yield read_depth, child_dir


def trial_dirs(read_depth_dir: Path) -> Iterator[Tuple[int, Path]]:
    for child_dir in read_depth_dir.glob("trial_*"):
        if not child_dir.is_dir():
            raise RuntimeError(f"Expected child `{child_dir}` to be a directory.")

        trial_num = int(child_dir.name.split("_")[1])
        yield trial_num, child_dir


def load_ground_truth(ground_truth_path: Path) -> pd.DataFrame:
    df_entries = []
    with open(ground_truth_path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='\"')
        header_row = next(reader)

        assert header_row[0] == 'T'
        strain_ids = header_row[1:]
        for row in reader:
            t = float(row[0])
            for strain_id, abund in zip(strain_ids, row[1:]):
                abund = float(abund)
                df_entries.append({'T': t, 'Strain': strain_id, 'RelAbund': abund})
    return pd.DataFrame(df_entries)


def error_metric(abundance_est: np.ndarray, ground_truth: pd.DataFrame) -> float:
    time_points = sorted(pd.unique(ground_truth['T']))
    strains = sorted(pd.unique(ground_truth['Strain']))
    ground_truth = np.array([
        [
            ground_truth.loc[(ground_truth['Strain'] == strain_id) & (ground_truth['T'] == t), 'RelAbund'].item()
            for strain_id in strains
        ]
        for t in time_points
    ])
    return np.sqrt(np.sum(
        np.square(abundance_est - ground_truth)
    ))


def parse_chronostrain_error(db: StrainDatabase, ground_truth: pd.DataFrame, output_dir: Path) -> float:
    abundance_samples = torch.load(output_dir / 'samples.pt')
    strains = db.all_strains()

    time_points = sorted(pd.unique(ground_truth['T']))

    if abundance_samples.shape[0] != len(time_points):
        raise RuntimeError("Number of time points ({}) in ground truth don't match sampled time points ({}).".format(
            len(time_points),
            abundance_samples.shape[0]
        ))

    if abundance_samples.shape[2] != len(strains):
        raise RuntimeError("Number of strains ({}) in database don't match sampled strain counts ({}).".format(
            len(strains),
            abundance_samples.shape[2]
        ))

    # hellingers = torch.square(
    #     torch.sqrt(samples, dim=2) - torch.unsqueeze(torch.sqrt(ground_truth_tensor), 1)
    # ).sum(dim=2).sqrt().mean(dim=0)  # Average hellinger distance across time, for each sample.
    # return torch.median(hellingers).item() / np.sqrt(2)
    median_abundances = np.median(abundance_samples.numpy(), axis=1)
    return error_metric(median_abundances, ground_truth)


def parse_straingst_error(ground_truth: pd.DataFrame, output_dir: Path, mode: str) -> float:
    time_points = sorted(pd.unique(ground_truth['T']))
    strains = sorted(pd.unique(ground_truth['Strain']))
    strain_indices = {strain_id: s_idx for s_idx, strain_id in enumerate(strains)}

    est_rel_abunds = np.zeros(shape=(len(time_points), len(strains)), dtype=float)

    for t_idx, t in enumerate(time_points):
        output_path = output_dir / f"output_{mode}_{t_idx}.tsv"
        if not output_path.exists():
            continue

        with open(output_path, 'r') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            _ = next(reader)
            _ = next(reader)
            line3 = next(reader)
            assert line3[0] == 'i'

            for row in reader:
                strain_id = row[1]
                strain_idx = strain_indices[strain_id]
                rel_abund = float(row[11]) / 100.0
                est_rel_abunds[t_idx][strain_idx] = rel_abund

    return error_metric(est_rel_abunds, ground_truth)


def parse_strainest_error(ground_truth: pd.DataFrame, output_dir: Path) -> float:
    time_points = sorted(pd.unique(ground_truth['T']))
    strains = sorted(pd.unique(ground_truth['Strain']))
    strain_indices = {strain_id: s_idx for s_idx, strain_id in enumerate(strains)}

    est_rel_abunds = np.zeros(shape=(len(time_points), len(strains)), dtype=float)
    for t_idx, t in enumerate(time_points):
        output_path = output_dir / f"abund_{t_idx}.txt"
        if not output_path.exists():
            continue

        with open(output_path, 'rt') as f:
            lines = iter(f)
            header_line = next(lines)
            if not header_line.startswith('OTU'):
                raise RuntimeError(f"Unexpected format for file `{output_path}` generated by StrainEst.")

            for line in lines:
                strain_id, abund = line.rstrip().split('\t')
                abund = float(abund)
                strain_idx = strain_indices[strain_id]
                est_rel_abunds[t_idx][strain_idx] = abund

    return error_metric(est_rel_abunds, ground_truth)


def get_baseline_diff(ground_truth: pd.DataFrame) -> float:
    time_points = sorted(pd.unique(ground_truth['T']))
    strains = sorted(pd.unique(ground_truth['Strain']))

    # baseline_arr = np.round(ground_truth, 0)
    baseline_arr = 0.5 * np.ones(shape=(len(time_points), len(strains)), dtype=float)
    return error_metric(baseline_arr, ground_truth)


def main():
    args = parse_args()

    base_dir = Path(args.base_data_dir)
    ground_truth = load_ground_truth(Path(args.ground_truth_path))
    db = cfg.database_cfg.get_database()
    out_path = Path(args.out_path)

    # search through all of the read depths.
    df_entries = []
    for read_depth, read_depth_dir in read_depth_dirs(base_dir):
        for trial_num, trial_dir in trial_dirs(read_depth_dir):
            print(f"Handling read depth {read_depth}, trial {trial_num}")

            # =========== Chronostrain
            chronostrain_err = parse_chronostrain_error(db, ground_truth, trial_dir / 'output' / 'chronostrain')
            df_entries.append({
                'ReadDepth': read_depth,
                'TrialNum': trial_num,
                'Method': 'Chronostrain',
                'Error': chronostrain_err
            })

            # =========== StrainGST
            straingst_mash_err = parse_straingst_error(ground_truth, trial_dir / 'output' / 'straingst', 'mash')
            straingst_fulldb_err = parse_straingst_error(ground_truth, trial_dir / 'output' / 'straingst', 'fulldb')
            df_entries.append({
                'ReadDepth': read_depth,
                'TrialNum': trial_num,
                'Method': 'StrainGST (mash)',
                'Error': straingst_mash_err
            })
            df_entries.append({
                'ReadDepth': read_depth,
                'TrialNum': trial_num,
                'Method': 'StrainGST (full DB)',
                'Error': straingst_fulldb_err
            })

            strainest_err = parse_strainest_error(ground_truth, trial_dir / 'output' / 'strainest')
            df_entries.append({
                'ReadDepth': read_depth,
                'TrialNum': trial_num,
                'Method': 'StrainEst',
                'Error': strainest_err
            })

    summary_df = pd.DataFrame(df_entries)
    summary_df.to_csv(out_path, index=False)
    print(f"[*] Saved results to {out_path}.")

    plot_path = out_path.parent / "plot.pdf"
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    sb.boxplot(
        data=summary_df,
        x='ReadDepth',
        hue='Method',
        y='Error',
        ax=ax
    )

    baseline_diff = get_baseline_diff(ground_truth)
    x = [-0.5, len(pd.unique(summary_df['ReadDepth'])) - 0.5]
    ax.plot(x, [baseline_diff, baseline_diff], linestyle='dotted', color='black', alpha=0.3)
    plt.savefig(plot_path)
    print(f"[*] Saved plot to {plot_path}.")


if __name__ == "__main__":
    main()
