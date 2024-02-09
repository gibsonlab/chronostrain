from typing import *
from pathlib import Path
import numpy as np


def load_ground_truth(mut_ratio: str, replicate: int) -> Tuple[List[str], List[float], np.ndarray]:
    ground_truth_path = Path(f"/mnt/e/semisynthetic_data/mutratio_{mut_ratio}/replicate_{replicate}/genomes/abundances.txt")
    with open(ground_truth_path, "rt") as f:
        slices = []
        time_points = []
        for line in f:
            line = line.strip()
            tokens = line.split(',')
            if tokens[0] == 'T':
                accessions = [
                    x[:-len(".READSIM_MUTANT")] if x.endswith(".READSIM_MUTANT") else x
                    for x in tokens[1:]
                ]
                continue

            time_points.append(float(tokens[0]))
            slices.append(np.array([float(x) for x in tokens[1:]]))
        return accessions, time_points, np.stack(slices)


def plot_ground_truth(ax, accessions, time_points, true_abundances, palette, label: bool = False):
    for i, acc in enumerate(accessions):
        c = palette[acc]
        traj = true_abundances[:, i]
        if label:
            ax.plot(time_points, traj, color=c, linestyle='--', marker='.', lw=1.5, label=acc)
        else:
            ax.plot(time_points, traj, color=c, linestyle='--', marker='.', lw=1.5)