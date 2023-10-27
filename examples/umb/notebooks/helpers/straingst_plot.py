from typing import *
from pathlib import Path
from dataclasses import dataclass
import datetime

import numpy as np
import pandas as pd
from Bio.Phylo.BaseTree import Tree
from matplotlib.axes import Axes
from strainge.io.utils import parse_straingst

from .tree import pruned_subtree, phylo_draw_custom


@dataclass
class StrainGSTUMBEntry:
    patient_id: str
    sample_name: str
    sample_type: str
    sra_id: str
    date: datetime.date
    time_point: float
    straingst_result_path: Path


def parse_umb_entries(straingst_output_basedir: Path, entries_csv_path: Path) -> List[StrainGSTUMBEntry]:
    entries: List[StrainGSTUMBEntry] = []
    df = pd.read_csv(entries_csv_path, sep=',')
    for _, row in df.iterrows():
        sample_type = row['type']
        patient_id = row['ID']
        sample_name = row['SampleName']
        sra_id = row['Run']

        if sample_type == 'stool':
            sample_type = 'Stool'
            straingst_path = straingst_output_basedir / 'stool' / 'output' / f'{patient_id}' / f'{sample_name}.tsv'
        elif sample_type == 'urine raw':
            sample_type = 'Urine'
            straingst_path = straingst_output_basedir / 'urine' / 'output' / f'{patient_id}' / f'{sample_name}.tsv'
        else:
            continue

        umb_date = datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()

        entries.append(StrainGSTUMBEntry(
            patient_id,
            sample_name,
            sample_type,
            sra_id,
            umb_date,
            row['days'],
            straingst_path
        ))

    # Parse plate scrape results.
    plate_output_dir = straingst_output_basedir / 'plate' / 'output'
    for fpath in plate_output_dir.glob("UMB*/*.tsv"):
        tokens = fpath.stem.split("_")
        assert tokens[0] == 'UMB'

        if len(tokens[1]) == 1:
            patient_id = f'UMB0{tokens[1]}'
            umb_stool_name = f'UMB0{tokens[1]}_{tokens[2]}'
        else:
            patient_id = f'UMB{tokens[1]}'
            umb_stool_name = f'UMB{tokens[1]}_{tokens[2]}'
        row = df.loc[df['SampleName'] == umb_stool_name].head(1)

        if row.shape[0] == 0:
            raise Exception(f"Couldn't find {patient_id}, sample {umb_stool_name}")

        sample_type = 'Plate'
        sample_name = fpath.stem

        sra_id = ''
        umb_date = datetime.datetime.strptime(row['date'].item(), '%Y-%m-%d').date()
        days = row['days'].item()
        entries.append(StrainGSTUMBEntry(
            patient_id,
            sample_name,
            sample_type,
            sra_id,
            umb_date,
            days,
            fpath
        ))
    return entries


def retrieve_patient_dates(umbs: List[StrainGSTUMBEntry]) -> pd.DataFrame:
    df_entries = []
    for umb_entry in umbs:
        df_entries.append({
            'Patient': umb_entry.patient_id,
            'SampleName': umb_entry.sample_name,
            'Date': umb_entry.date,
            'T': umb_entry.time_point,
            'Src': umb_entry.sample_type
        })
    return pd.DataFrame(df_entries).astype(
        dtype={
            "Patient": "object",
            "SampleName": "object",
            "Date": "datetime64[ns]",
            "T": "float64"
        }
    )


def fetch_strain_id_from_straingst(strain_name: str, index_df: pd.DataFrame) -> Tuple[str, str, str]:
    fasta_path = Path("/mnt/e/strainge/strainge_db") / strain_name
    gcf_id = '_'.join(fasta_path.resolve().stem.split('_')[:2])
    hit = index_df.loc[index_df['Assembly'] == gcf_id, :].head(1)
    if hit.shape[0] == 0:
        raise ValueError(f"Couldn't find strain from StrainGST identifier `{strain_name}`.")
    return hit['Accession'].item(), hit['Strain'].item(), hit['Species'].item()


def straingst_dataframe(umb_entries: List[StrainGSTUMBEntry], phylogroup_path: Path) -> pd.DataFrame:
    index_df = pd.read_csv("/mnt/e/ecoli_db/ref_genomes/index.tsv", sep='\t')
    phylogroup_mappings = {}
    with open(phylogroup_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            accession = Path(tokens[0]).with_suffix('').with_suffix('').name
            phylogroup = tokens[4]
            phylogroup_mappings[accession] = phylogroup

    df_entries = []
    for umb_entry in umb_entries:
        if not umb_entry.straingst_result_path.exists():
            continue
        with open(umb_entry.straingst_result_path, "r") as f:
            for strain in parse_straingst(f):
                try:
                    strain_tag = strain['strain']
                    if not (strain_tag.startswith('Esch') or strain_tag.startswith('Shig')):
                        print(f"UMB entry `{umb_entry.sample_name}` contains non-Esch/Shig name `{strain_tag}`.")
                        continue
                    accession, strain_name, species_name = fetch_strain_id_from_straingst(strain_tag, index_df)
                except ValueError as e:
                    raise ValueError(f"Problem while loading {umb_entry.sample_name}")
                phylogroup = phylogroup_mappings.get(accession, "N/A")

                df_entries.append({
                    'Patient': umb_entry.patient_id,
                    'SampleName': umb_entry.sample_name,
                    'Date': umb_entry.date,
                    'StrainName': strain_name,
                    'Species': species_name,
                    'StrainId': accession,
                    'Phylogroup': phylogroup,
                    'RelAbund': float(strain['rapct']) / 100.0
                })

    return pd.DataFrame(df_entries).astype(
        dtype={
            "Date": "datetime64[ns]",
            "RelAbund": "float64"
        }
    )


# ===================================== Plotting
def plot_straingst_abundances(
        strain_df: pd.DataFrame,
        dates_df: pd.DataFrame,
        clade_colors: Dict[str, np.ndarray],
        ax: Axes,
        mode: str = 'stool',
        yscale: str = 'log'
):
    """
    returns a mapping of leaf node names to y-axis positions, as well as the appropriate ylim tuple.
    strain_df: A dataframe with the following columns: T, StrainId, Phylogroup, RelAbund, Src, Date
    dates_df: A dataframe with column 'Date', filled with all dates relevant to the patient-sampletype being plotted (e.g. umb18-stool).
    """
    if mode == 'stool':
        dates_df = dates_df.loc[dates_df['Src'] == 'Stool']
    elif mode == 'urine':
        dates_df = dates_df.loc[dates_df['Src'] == 'Urine']
    else:
        raise ValueError(f"Unknown plotting mode `{mode}`.")

    for strain, strain_group in strain_df.groupby('StrainId'):
        clade = list(pd.unique(strain_group['Phylogroup']))[0]

        try:
            color = clade_colors[clade]
        except KeyError:
            print(f"WARNING: found unrecognized clade `{clade}` for strain {strain}.")
            color = "black"

        strain_group = strain_group[['SampleName', 'RelAbund']].merge(
            dates_df, on='SampleName', how='right'
        ).fillna(1e-50).sort_values('T')
        ax.plot(strain_group['T'], strain_group['RelAbund'], marker='.', linewidth=2, color=color)
    ax.set_yscale(yscale)


def plot_tree(
        strain_df: pd.DataFrame,
        clade_colors: Dict[str, np.ndarray],
        ax: Axes,
        tree: Tree,
) -> Tuple[Dict, Dict]:
    strain_id_to_names = {}
    strain_id_to_colors = {}
    for _, row in strain_df.iterrows():
        strain_id = row['StrainId']
        strain_name = row['StrainName']
        phylogroup = row['Phylogroup']

        strain_id_to_names[strain_id] = strain_name
        strain_id_to_colors[strain_id] = clade_colors[phylogroup]

    strain_leaves = set(strain_id_to_names.keys())
    if len(strain_leaves) == 0:
        return {}, {}

    subtree = pruned_subtree(tree, strain_leaves)

    def color_fn(s):
        if s.is_terminal():
            return strain_id_to_colors[s.name]
        else:
            return "black"

    def label_fn(s):
        if s.is_terminal():
            return "{}".format(strain_id_to_names[s.name])
        else:
            return ""

    subtree.name = ''
    x_posns, y_posns = phylo_draw_custom(
        subtree,
        label_func=label_fn,
        axes=ax,
        do_show=False,
        show_confidence=False,
        label_colors=color_fn,
        branch_labels=lambda c: '{:.03f}'.format(c.branch_length) if (
                    c.branch_length is not None and c.branch_length > 0.003) else ''
    )
    ax.axis('off')
    return x_posns, y_posns


def plot_clade_presence(
        strain_df: pd.DataFrame,
        dates_df: pd.DataFrame,
        clade_colors: Dict[str, np.ndarray],
        ax: Axes,
        strain_y: Optional[Dict[str, int]] = None,
        show_ylabels: bool = True
):
    if strain_df.shape[0] == 0:
        return
    if strain_y is None:
        _ids = sorted(pd.unique(strain_df['StrainId']))
        strain_y = {_id: _i for _i, _id in enumerate(_ids)}
    strain_df = strain_df.assign(Y=strain_df['StrainId'].map(strain_y))
    strain_df = strain_df.merge(
        dates_df,
        on=['Patient', 'SampleName', 'Date'],
        how='inner'
    )

    marker_sz = 1.0
    for (src, strain_id, phylogroup), section in strain_df.groupby(["Src", "StrainId", "Phylogroup"]):
        color = clade_colors[phylogroup]

        # Pick style
        if src == 'Stool':
            ax.scatter(section['T'], section['Y'], edgecolor=color, facecolors=[1, 1, 1, 0], marker='o',
                       linewidths=2 * marker_sz, s=200 * marker_sz, zorder=2)
        elif src == 'Plate':
            ax.scatter(section['T'], section['Y'], facecolors=color, marker='x', linewidths=1.5 * marker_sz,
                       s=100 * marker_sz, zorder=2)
        elif src == 'Urine':
            ax.scatter(section['T'], section['Y'], edgecolor=color, facecolors=color, marker='.',
                       linewidths=2 * marker_sz, s=200 * marker_sz, zorder=2)

    time_points = sorted(strain_df['T'])
    ax.set_yticks(sorted(strain_y.values()))
    ax.set_xticks(time_points)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # ======= Other settings.
    y_min = strain_df['Y'].min() - 1
    y_max = strain_df['Y'].max() + 1
    ax.set_ylim(bottom=y_min, top=y_max)

    if show_ylabels:
        labels = []
        for y, _df in strain_df.sort_values('Y').groupby('Y'):
            labels.append(_df.head(1)['StrainName'].item())
        ax.set_yticklabels(labels=labels)
    else:
        ax.set_yticklabels(labels=["" for _ in range(len(strain_y))])

