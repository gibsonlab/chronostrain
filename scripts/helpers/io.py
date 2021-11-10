import csv
from pathlib import Path
from typing import Dict, Tuple, List

from chronostrain.model.io import TimeSeriesReads, TimeSliceReadSource


def parse_reads(input_spec_path: Path, quality_format: str):
    read_sources, read_depths, time_points = parse_input_spec(input_spec_path, quality_format)

    return TimeSeriesReads.load(
        time_points=time_points,
        read_depths=read_depths,
        sources=read_sources
    )


def parse_input_spec(input_spec_path: Path, quality_format: str) -> Tuple[List[TimeSliceReadSource], List[int], List[float]]:
    time_points_to_reads: Dict[float, List[Tuple[int, Path]]] = {}
    if not input_spec_path.exists():
        raise FileNotFoundError(f"Missing required file `{str(input_spec_path)}`")

    with open(input_spec_path, "r") as f:
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

    time_points = sorted(time_points_to_reads.keys(), reverse=False)
    read_depths = [
        sum([
            n_reads for n_reads, _ in time_points_to_reads[t]
        ])
        for t in time_points
    ]

    time_slice_sources = [
        TimeSliceReadSource([
            read_path for _, read_path in time_points_to_reads[t]
        ], quality_format)
        for t in time_points
    ]

    return time_slice_sources, read_depths, time_points
