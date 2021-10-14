import csv
from pathlib import Path
from typing import Dict, Tuple, List, Iterable


def get_input_paths(base_dir: Path, input_filename) -> Tuple[List[Iterable[Path]], List[int], List[float]]:
    time_points_to_reads: Dict[float, List[Tuple[int, Path]]] = {}

    input_specification_path = base_dir / input_filename
    try:
        with open(input_specification_path, "r") as f:
            input_specs = csv.reader(f, delimiter=',', quotechar='"')
            for row in input_specs:
                time_point = float(row[0])
                num_reads = int(row[1])
                read_path = Path(row[2])

                if not read_path.exists():
                    raise FileNotFoundError(f"Specified input file `{str(read_path)}` does not exist.")

                if time_point not in time_points_to_reads:
                    time_points_to_reads[time_point] = []

                time_points_to_reads[time_point].append((num_reads, read_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing required file `{input_filename}` in directory {base_dir}.") from None

    time_points = sorted(time_points_to_reads.keys(), reverse=False)
    read_depths = [
        sum([
            n_reads for n_reads, _ in time_points_to_reads[t]
        ])
        for t in time_points
    ]
    read_paths = [
        [
            read_path for _, read_path in time_points_to_reads[t]
        ]
        for t in time_points
    ]

    return read_paths, read_depths, time_points
