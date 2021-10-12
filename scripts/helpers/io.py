import csv
from pathlib import Path
from typing import Tuple, List, Iterable


def get_input_paths(base_dir: Path, input_filename) -> Tuple[List[Iterable[Path]], List[int], List[float]]:
    time_points: List[float] = []
    read_depths: List[int] = []
    read_paths: List[List[Path]] = []

    input_specification_path = base_dir / input_filename
    try:
        with open(input_specification_path, "r") as f:
            input_specs = csv.reader(f, delimiter=',', quotechar='"')
            for row in input_specs:
                time_point_str = row[0]
                read_depth_str = row[1]
                filenames = [base_dir / f for f in row[2:]]

                time_points.append(float(time_point_str))
                read_depths.append(int(read_depth_str))
                read_paths.append(filenames)
    except FileNotFoundError:
        raise FileNotFoundError("Missing required file `input_files.csv` in directory {}.".format(base_dir)) from None

    if len(read_paths) != len(time_points):
        raise ValueError("There must be exactly one set of reads for each time point specified.")

    if len(read_depths) != len(time_points):
        raise ValueError("Input specification must include a read depth for each time point.")

    if len(time_points) != len(set(time_points)):
        raise ValueError("Specified sample times must be distinct.")

    for read_paths_t in read_paths:
        for f in read_paths_t:
            if not f.exists():
                raise FileNotFoundError("Specified input file `{}` does not exist.".format(str(f)))

    return read_paths, read_depths, time_points
