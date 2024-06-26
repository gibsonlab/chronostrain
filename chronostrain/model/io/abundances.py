import csv
from pathlib import Path
from typing import List, Tuple

import jax.numpy as np
from chronostrain.model import StrainCollection
from chronostrain.util.filesystem import convert_size

from chronostrain.logging import create_logger
logger = create_logger(__name__)


def save_abundances(
        population: StrainCollection,
        time_points: List[float],
        abundances: np.ndarray,
        out_path: Path):
    """
    Save the time-indexed abundance profile to disk. Output format is CSV.

    :param population: The Population instance containing the strain information.
    :param time_points: The list of time points in the read_frags.
    :param abundances: A T x S tensor containing time-indexed relative abundances profiles.
    :param out_path: The target path for the output abundances file.
    :return: The path/filename for the abundance CSV file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow(["T"] + [strain.id for strain in population.strains])
        for t in range(len(time_points)):
            writer.writerow([time_points[t]] + [x.item() for x in abundances[t]])
    logger.info("Abundances output successfully to {}. ({})".format(
        out_path, convert_size(out_path.stat().st_size)
    ))


def load_abundances(file_path: Path) -> Tuple[List[int], np.ndarray, List[str]]:
    """
    Read time-indexed abundances from file.

    :return: (1) A list of time points,
    (2) a time indexed list of abundance profiles,
    (3) the list of relevant accessions.
    """
    time_points = []
    strain_abundances = []
    accessions = []

    with open(file_path, newline='') as f:
        reader = csv.reader(f, quotechar='"')
        for i, row in enumerate(reader):
            if i == 0:
                accessions = [x.replace('"', '').strip() for x in row[1:]]
                continue
            if not row:
                continue
            time_point = row[0]
            abundances = np.array([float(val) for val in row[1:]])
            time_points.append(time_point)
            strain_abundances.append(abundances)
    return time_points, np.stack(strain_abundances, axis=0), accessions
