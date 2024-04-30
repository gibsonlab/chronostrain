from typing import List
from pathlib import Path
import pandas as pd


def get_infant_ids(participants_file: Path):
    with open(participants_file, "rt") as f:
        for line in f:
            infant_id = line.rstrip()
            yield infant_id


def load_isolate_metadata(infant_id: str, data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / infant_id / 'isolate_assemblies' / 'metadata.tsv'
    return pd.read_csv(metadata_path, sep='\t')


def load_isolate_fastmlst(infant_id: str, data_dir: Path) -> pd.DataFrame:
    # Load FastMLST annotations.
    fastmlst_output = data_dir / infant_id / 'isolate_assemblies' / 'fastmlst.tsv'
    df_entries = []
    with open(fastmlst_output, "rt") as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue
            tokens = line.rstrip().split('\t')
            acc = tokens[0]
            if acc.endswith('.fasta'):
                acc = acc[:-6]
            tax_str = tokens[1]
            st = tokens[2]
            df_entries.append(
                (acc, f'{tax_str}:{st}')
            )
    return pd.DataFrame(
        df_entries,
        columns=['Accession', 'ST']
    )


def load_all_isolate_metadata(infant_ids: List[str], data_dir: Path) -> pd.DataFrame:
    sections = []
    for infant_id in infant_ids:
        sections.append(
            load_isolate_metadata(infant_id, data_dir)
        )
    return pd.concat(sections, ignore_index=True)


def load_all_isolate_metadata_with_fastmlst(infant_ids: List[str], data_dir: Path) -> pd.DataFrame:
    sections = []
    for infant_id in infant_ids:
        sections.append(
            load_isolate_metadata(infant_id, data_dir).merge(
                load_isolate_fastmlst(infant_id, data_dir),
                on='Accession',
                how='outer'
            )
        )
    return pd.concat(sections, ignore_index=True)

