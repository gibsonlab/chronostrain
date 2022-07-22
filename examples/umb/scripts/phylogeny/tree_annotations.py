import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.colors as mc
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Output annotation files for the tree generated for strains in database."
    )

    # Input specification.
    parser.add_argument('-o', '--output_dir', required=True, type=str,
                        help='<Required> The target output directory.')
    parser.add_argument('-i', '--index', dest='strain_index_path',
                        required=True, type=str,
                        help='<Required> The path to the TSV refseq index file.')
    parser.add_argument('-p', '--phylogroup_path', required=True, type=str,
                        help='<Required> The path to ClermonTyping phylogroup output txt file.')
    return parser.parse_args()


def parse_strain_index(index_path: Path) -> List[Tuple[str, str]]:
    """
    Return a list of (strain accession, strain name) tuples.
    """
    df = pd.read_csv(index_path, sep='\t')
    strains = []
    for _, row in df.loc[df['Genus'] == 'Escherichia'].iterrows():
        strains.append((
            row['Accession'], row['Strain']
        ))
    return strains


def create_clade_annotation(target_path: Path,
                            strains: List[Tuple[str, str]],
                            strain_clades: Dict[str, str],
                            clade_hex_colors: Dict[str, str]):
    with open(target_path, 'w') as f:
        print("TREE_COLORS", file=f)
        print("SEPARATOR COMMA", file=f)
        print("DATA", file=f)
        print("#NODE_ID,TYPE,COLOR,LABEL_OR_STYLE,SIZE_FACTOR", file=f)
        for strain_id, _ in strains:
            clade = strain_clades[strain_id]
            color = clade_hex_colors[clade]
            print(f"{strain_id},range,{color},{clade}", file=f)


def create_label_annotation(target_path: Path, strains: List[Tuple[str, str]]):
    with open(target_path, 'w') as f:
        print("LABELS", file=f)
        print("SEPARATOR COMMA", file=f)
        print("DATA", file=f)
        for strain_id, strain_name in strains:
            print(f"{strain_id},{strain_name}", file=f)


def load_clade_assignments(phylogroup_path: Path) -> Dict[str, str]:
    strain_to_phylogroup = {}
    with open(phylogroup_path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            accession = Path(tokens[0]).with_suffix('').with_suffix('').name
            phylogroup = tokens[4]
            strain_to_phylogroup[accession] = phylogroup
    return strain_to_phylogroup


def load_clade_colors() -> Dict[str, str]:
    colors = sns.color_palette("tab20")
    palette: Dict[str, str] = {
        phylo_grp: mc.to_hex(colors[p_idx])
        for p_idx, phylo_grp in enumerate([
            "A", "B1", "B2", "C", "D", "E", "F", "G", "fergusonii",
            "albertii", "E or cladeI", "cladeI", "Unknown", "cladeV"
        ])
    }
    return palette


def main():
    args = parse_args()

    strains = parse_strain_index(Path(args.strain_index_path))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    clades = load_clade_assignments(Path(args.phylogroup_path))
    clade_colors = load_clade_colors()

    create_label_annotation(out_dir / "labels.txt", strains)
    create_clade_annotation(out_dir / "clades.txt", strains, clades, clade_colors)


if __name__ == "__main__":
    main()
