import argparse
from pathlib import Path
from typing import Tuple, Iterator

import csv
import pandas as pd
import torch
import matplotlib
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


def parse_chronostrain_error(db: StrainDatabase, ground_truth: pd.DataFrame, output_dir: Path) -> torch.Tensor:
    samples = torch.load(output_dir / 'samples.pt')
    strains = db.all_strains()

    time_points = sorted(pd.unique(ground_truth['T']))

    if samples.shape[0] != len(time_points):
        raise RuntimeError("Number of time points ({}) in ground truth don't match sampled time points ({}).".format(
            len(time_points),
            samples.shape[0]
        ))

    if samples.shape[2] != len(strains):
        raise RuntimeError("Number of strains ({}) in database don't match sampled strain counts ({}).".format(
            len(strains),
            samples.shape[2]
        ))

    # ground truth array
    ground_truth_tensor = torch.tensor([
        [
            ground_truth.loc[(ground_truth['Strain'] == strain.id) & (ground_truth['T'] == t), 'RelAbund'].item()
            for strain in strains
        ]
        for t in time_points
    ])

    hellingers = torch.square(
        torch.sqrt(torch.softmax(samples, dim=2)) - torch.unsqueeze(torch.sqrt(ground_truth_tensor), 1)
    ).sum(dim=2).sqrt().mean(dim=0)  # Average hellinger distance across time, for each sample.
    return torch.median(hellingers).item()


def main():
    args = parse_args()

    base_dir = Path(args.base_data_dir)
    ground_truth = load_ground_truth(Path(args.ground_truth_path))
    db = cfg.database_cfg.get_database()
    out_path = Path(args.out_path)

    # search through all of the read depths.
    df_entries = []
    print("[*] WARNING: TODO: implement straingst parsing.")
    for read_depth, read_depth_dir in read_depth_dirs(base_dir):
        for trial_num, trial_dir in trial_dirs(read_depth_dir):
            print(f"Handling read depth {read_depth}, trial {trial_num}")
            chronostrain_hellinger = parse_chronostrain_error(
                db,
                ground_truth,
                trial_dir / 'output' / 'chronostrain'
            )

            df_entries.append({
                'ReadDepth': read_depth,
                'TrialNum': trial_num,
                'Method': 'Chronostrain',
                'Error': chronostrain_hellinger
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
    plt.savefig(plot_path)
    print(f"[*] Saved plot to {plot_path}.")


if __name__ == "__main__":
    main()