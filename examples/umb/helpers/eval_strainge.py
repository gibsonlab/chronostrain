from typing import Tuple, Iterator, Dict, List
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
    parser.add_argument('-r', '--refseq_index', required=True, type=str)

    parser.add_argument('--group_by_clades', action='store_true')
    parser.add_argument('-c', '--clades', required=False, type=str)
    return parser.parse_args()


def strip_suffixes(x):
    x = Path(x)
    suffix_set = {'.chrom', 'fa', '.fna', '.gz', 'fasta'}
    while x.suffix in suffix_set:
        x = x.with_suffix('')
    return x.name


def fetch_strain_id(strain_name: str, ref_df: pd.DataFrame) -> str:
    hits = ref_df.loc[ref_df['Strain'] == strain_name, 'Accession']
    if hits.shape[0] == 0:
        raise RuntimeError("Unknown strain name `{strain_name}` found.")
    if hits.shape[0] > 1:
        raise RuntimeError("Ambiguous strain name `{strain_name}`.")
    return hits.item()


def parse_clades(clades_path: Path, ref_df: pd.DataFrame) -> Dict[str, str]:
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
            strain_name = strip_suffixes(tokens[0])
            strain_id = fetch_strain_id(strain_name, ref_df)
            phylogroup = tokens[4]
            mapping[strain_id] = phylogroup
    return mapping


def parse_outputs(base_dir: Path) -> Iterator[Tuple[str, pd.DataFrame]]:
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
            strain_id = strip_suffixes(strain['strain'])
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


def convert_to_numpy(timeseries_df: pd.DataFrame, metadata: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
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
    return timeseries, strains


def evaluate(strainge_output_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    df_entries = []
    for patient, timeseries_df in parse_outputs(strainge_output_dir):
        timeseries, _ = convert_to_numpy(timeseries_df, metadata)
        df_entries.append({
            "Patient": patient,
            "Dominance": dominance_switch_ratio(timeseries)
        })
    return pd.DataFrame(df_entries)


def evaluate_by_clades(strainge_output_dir: Path, clades: Dict[str, str], metadata: pd.DataFrame) -> pd.DataFrame:
    df_entries = []
    for patient, timeseries_df in parse_outputs(strainge_output_dir):
        timeseries, strain_ids = convert_to_numpy(timeseries_df, metadata)
        for clade, sub_timeseries in divide_into_timeseries(timeseries, strain_ids, clades):
            df_entries.append({
                "Patient": patient,
                "Phylogroup": clade,
                "Dominance": dominance_switch_ratio(sub_timeseries)
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
    ref_df = pd.read_csv(args.refseq_index)

    if args.group_by_clades:
        if args.clades is None:
            print("If grouping by clades, a clades path is required.")
            exit(1)
        clades = parse_clades(args.clades, ref_df)
        df = evaluate_by_clades(Path(args.strainge_dir), clades, metadata)
    else:
        df = evaluate(Path(args.strainge_dir), metadata)

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
