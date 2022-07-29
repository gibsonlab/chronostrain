from typing import Tuple, Iterator
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from strainge.io.utils import parse_straingst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--strainge_dir', required=True, type=str)
    parser.add_argument('-m', '--metadata', required=True, type=str)
    parser.add_argument('-o', '--output', required=True, type=str)
    return parser.parse_args()


def parse_outputs(base_dir: Path) -> Iterator[pd.DataFrame]:
    entries = []
    for umb_id, umb_dir in umb_dirs(base_dir):
        print(f"Handling {umb_id}.")
        for sample_id, output_file in output_files(umb_dir):
            print(f"Reading output file {output_file}.")
            for strain_id, rel_abund in parse_single_output(output_file):
                entries.append({
                    'Sample': sample_id,
                    'Strain': strain_id,
                    'RelAbund': rel_abund
                })
        yield umb_id, pd.DataFrame(entries)


def parse_single_output(output_file: Path) -> Iterator[Tuple[str, float]]:
    with open(output_file, "r") as f:
        for strain in parse_straingst(f):
            strain_id = strain['strain']
            rel_abund = float(strain['rapct']) / 100.0
            yield strain_id, rel_abund


def umb_dirs(base_dir: Path) -> Iterator[Tuple[str, Path]]:
    for umb_dir in base_dir.glob("UMB*"):
        if not umb_dir.is_dir():
            raise RuntimeError(f"Expected child `{umb_dir}` to be a directory.")
        umb_id = umb_dir.name
        yield umb_id, umb_dir


def output_files(patient_dir: Path):
    for output_file in patient_dir.glob("*.tsv"):
        sample_id = output_file.with_suffix('').name
        yield sample_id, output_file


def convert_to_numpy(timeseries_df: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
    """
    Run,ID,SampleName,date,days,type,Model,LibraryStrategy,Group
    SRR14881730,UMB01,UMB01_00,2015-10-26,298,stool,HiSeq X Ten,WGS,Control
    """
    merged = timeseries_df.merge(
        metadata[['SampleName', 'date', 'days']],
        left_on='Sample',
        right_on='SampleName',
        how='left'
    )

    time_points = list(pd.unique(merged['days']))
    strains = list(pd.unique(timeseries_df['Strain']))

    time_indexes = {t: i for i, t in enumerate(time_points)}
    strain_indexes = {s: i for i, s in enumerate(strains)}
    timeseries = np.zeros((len(time_points), len(strains)), dtype=float)

    for _, row in merged.iterrows():
        day = row['days']
        strain = row['Strain']
        tidx = time_indexes[day]
        sidx = strain_indexes[strain]
        timeseries[tidx, sidx] = row['RelAbund']
    return timeseries


def evaluate(strainge_output_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    df_entries = []
    for patient, timeseries_df in parse_outputs(strainge_output_dir):
        timeseries = convert_to_numpy(timeseries_df, metadata)
        df_entries.append({
            "Patient": patient,
            "Dominance": dominance_switch_ratio(timeseries)
        })
    return pd.DataFrame(df_entries)


def dominance_switch_ratio(abundance_est: np.ndarray) -> float:
    """
    Calculate how often the dominant strain switches.
    """
    dom = np.argmax(abundance_est, axis=1)
    num_switches = 0

    def row_is_zeros(r) -> bool:
        return np.sum(r == 0).item() == r.shape[0]

    for i in range(len(dom) - 1):
        switched = (dom[i] != dom[i+1]) or row_is_zeros(abundance_est[i]) or row_is_zeros(abundance_est[i+1])
        if switched:
            num_switches += 1
    return num_switches / (len(dom) - 1)


def main():
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    df = evaluate(Path(args.strainge_dir), metadata)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
