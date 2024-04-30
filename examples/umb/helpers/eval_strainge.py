from typing import Tuple, Iterator, Dict, List
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import scipy.stats

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


def fetch_strain_id_from_straingst(strain_name: str, index_df: pd.DataFrame) -> Tuple[str, str]:
    fasta_path = Path("/mnt/e/strainge/strainge_db") / strain_name
    gcf_id = '_'.join(fasta_path.resolve().stem.split('_')[:2])
    hit = index_df.loc[index_df['Assembly'] == gcf_id, :].head(1)
    if hit.shape[0] == 0:
        raise ValueError(f"Couldn't find strain from StrainGST identifier `{strain_name}`.")
    return hit['Accession'].item(), hit['Strain'].item()


def strip_suffixes(x: str) -> str:
    x = Path(x)
    suffix_set = {'.chrom', '.fa', '.fna', '.gz', 'fasta'}
    while x.suffix in suffix_set:
        x = x.with_suffix('')
    return x.name


# def strip_prefix(x: str):
#     return "_".join(x.split("_")[2:])
#
#
# class StrainNotFound(BaseException):
#     pass
#
#
# def fetch_strain_id(strain_name: str, ref_df: pd.DataFrame) -> str:
#     # preprocess.
#     strain_name = strip_suffixes(strain_name)
#     strain_name = strip_prefix(strain_name)
#
#     if "GCF" in strain_name:
#         tokens = strain_name.split("_GCF_")
#         strain_name = tokens[0]
#         gcf_id = "GCF_{}".format(tokens[1])
#         ref_df = ref_df.loc[ref_df['Assembly'] == gcf_id, :]
#
#     strain_names_to_try = [strain_name, strain_name.replace("_", ".")]
#     if strain_name.startswith("str"):
#         short_name = "_".join(strain_name.split("_")[1:])
#         strain_names_to_try.append(short_name)
#         strain_names_to_try.append(f"str._{short_name}")
#         strain_names_to_try.append(f"Escherichia_coli_str._{short_name}")
#
#     for s in strain_names_to_try:
#         try:
#             return search_df(s, ref_df)
#         except StrainNotFound:
#             print(f"Unable to find strain name entry `{s}`. Remaining possibilities: {strain_names_to_try}")
#     raise RuntimeError(f"Unknown strain name `{strain_name}` encountered.")
#
#
# def search_df(strain_name: str, ref_df: pd.DataFrame):
#     hits = ref_df.loc[ref_df['Strain'] == strain_name, 'Accession']
#     if hits.shape[0] == 0:
#         raise StrainNotFound(f"Unknown strain name `{strain_name}` encountered.")
#
#     result = hits.head(1).item()
#     if hits.shape[0] > 1:
#         print(f"Ambiguous strain name `{strain_name}`. Taking the first accession {result}")
#     return result


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


def parse_outputs(base_dir: Path, ref_df: pd.DataFrame) -> Iterator[Tuple[str, pd.DataFrame]]:
    for umb_id, umb_dir in umb_dirs(base_dir):
        entries = []
        print(f"Handling {umb_id}.")
        for sample_id, output_file in output_files(umb_dir):
            print(f"Reading output file {output_file}.")
            for strain_id, rel_abund in parse_single_output(output_file, ref_df):
                entries.append({
                    'Sample': sample_id,
                    'Strain': strain_id,
                    'RelAbund': rel_abund
                })
        yield umb_id, pd.DataFrame(entries)


def parse_single_output(output_file: Path, ref_df: pd.DataFrame) -> Iterator[Tuple[str, float]]:
    with open(output_file, "r") as f:
        for strain in parse_straingst(f):
            strain_name = strain['strain']
            try:
                strain_id, _ = fetch_strain_id_from_straingst(strain_name, ref_df)
            except ValueError:
                print("Couldn't identify an accession number for `{strain_name}` in the StrainGE database.")
                continue
            rel_abund = float(strain['rapct']) / 100.0
            yield strain_id, rel_abund


def umb_dirs(base_dir: Path) -> Iterator[Tuple[str, Path]]:
    print(f"Searching for UMB results in {base_dir}")
    for umb_dir in base_dir.glob("UMB*"):
        if not umb_dir.is_dir():
            raise RuntimeError(f"Expected child `{umb_dir}` to be a directory.")
        print(f"Navigating {umb_dir}")
        umb_id = umb_dir.name
        yield umb_id, umb_dir


def output_files(patient_dir: Path):
    for output_file in patient_dir.glob("*.tsv"):
        sample_id = output_file.with_suffix('').name
        yield sample_id, output_file


def convert_to_numpy(timeseries_df: pd.DataFrame, patient: str, metadata: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
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

    time_points = list(pd.unique(
        metadata.loc[
            (metadata['ID'] == patient) & (metadata['type'] == 'stool'),
            'days'
        ]
    ))
    strains = list(pd.unique(timeseries_df['Strain']))

    time_indexes = {t: i for i, t in enumerate(time_points)}
    strain_indexes = {s: i for i, s in enumerate(strains)}
    timeseries = np.zeros((len(time_points), len(strains)), dtype=float)

    try:
        for _, row in merged.iterrows():
            day = row['days']
            strain = row['Strain']
            tidx = time_indexes[day]
            sidx = strain_indexes[strain]
            timeseries[tidx, sidx] = row['RelAbund']
    except KeyError as e:
        merged.to_csv(f"__{patient}_DATA_DUMP.csv", index=False)
        raise e

    return timeseries, strains


def evaluate(strainge_output_dir: Path, metadata: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    df_entries = []
    for patient, timeseries_df in parse_outputs(strainge_output_dir, ref_df):
        if timeseries_df.shape[0] == 0:
            df_entries.append({
                "Patient": patient,
                "Coherence": np.nan
            })
        else:
            timeseries, _ = convert_to_numpy(timeseries_df, patient, metadata)
            df_entries.append({
                "Patient": patient,
                "Coherence": timeseries_coherence_factor(timeseries)
            })
    return pd.DataFrame(df_entries)


def evaluate_by_clades(strainge_output_dir: Path, clades: Dict[str, str], metadata: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    df_entries = []
    for patient, timeseries_df in parse_outputs(strainge_output_dir, ref_df):
        if timeseries_df.shape[0] == 0:
            for clade in set(clades.values()):
                df_entries.append({
                    "Patient": patient,
                    "Phylogroup": clade,
                    "Coherence": np.nan,
                    "OverallRelAbundMax": np.nan,
                    "StrainRelAbundMax": np.nan
                })
        else:
            timeseries, strain_ids = convert_to_numpy(timeseries_df, patient, metadata)
            for clade, sub_timeseries in divide_into_timeseries(timeseries, strain_ids, clades):
                print("CLADE: {}".format(clade))
                if sub_timeseries.shape[1] == 0:
                    df_entries.append({
                        "Patient": patient,
                        "Phylogroup": clade,
                        "Coherence": np.nan,
                        "OverallRelAbundMax": np.nan,
                        "StrainRelAbundMax": np.nan,
                    })
                else:
                    df_entries.append({
                        "Patient": patient,
                        "Phylogroup": clade,
                        "Coherence": timeseries_coherence_factor(sub_timeseries),
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
            yield this_clade, np.empty((timeseries.shape[0], 0))
        else:
            yield this_clade, timeseries[:, matching_strains]


def timeseries_coherence_factor(x: np.ndarray) -> np.ndarray:
    print("DATA:")
    print(x)
    _ans = mean_coherence_factor(x[1:], x[:-1])
    print("ANS")
    print(_ans)
    return _ans


def mean_coherence_factor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.shape[0] == y.shape[0]
    assert x.shape[-1] == y.shape[-1]

    if len(x.shape) == 2 and len(y.shape) == 2:
        print("ASDF")
        print(
            np.array([
                coherence_factor(x_t, y_t)
                for x_t, y_t in zip(x, y)
            ])
        )
        return np.nanmean([
            coherence_factor(x_t, y_t)
            for x_t, y_t in zip(x, y)
        ])

    if len(x.shape) == 2:
        x = np.repeat(np.expand_dims(x, 1), y.shape[1], axis=1)
    if len(y.shape) == 2:
        y = np.repeat(np.expand_dims(y, 1), x.shape[1], axis=1)

    return np.nanmean([
        [
            coherence_factor(x_tn, y_tn)
            for x_tn, y_tn in zip(x_t, y_t)
        ]
        for x_t, y_t in zip(x, y)
    ], axis=0)


def coherence_factor(x: np.ndarray, y: np.ndarray) -> float:
    assert len(x.shape) == 1
    assert len(y.shape) == 1

    if np.std(x) == 0 and np.std(y) == 0:  # edge case
        if x[0] == 0.:
            return np.nan  # zero abundance -> nothing to evaluate.
        elif x[0] == y[0]:
            return 1.0
        else:
            return 1.0

    if np.std(x) == 0 or np.std(y) == 0:  # only one is zero
        return 0.0

    # noinspection PyTypeChecker
    return scipy.stats.spearmanr(x, y)[0]


def main():
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    ref_df = pd.read_csv(args.refseq_index, sep='\t')

    if args.group_by_clades:
        if args.clades is None:
            print("If grouping by clades, a clades path is required.")
            exit(1)
        clades = parse_clades(args.clades)
        df = evaluate_by_clades(Path(args.strainge_dir), clades, metadata, ref_df)
    else:
        df = evaluate(Path(args.strainge_dir), metadata, ref_df)

    df.to_csv(args.output, index=False, sep='\t')


if __name__ == "__main__":
    main()
