import argparse
from pathlib import Path
from typing import *

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index_path', type=str, required=True)
    parser.add_argument('-s', '--strainge_refs_to_keep', type=str, required=True)
    parser.add_argument('-m', '--strainge_refs_meta', type=str, required=True)
    parser.add_argument('-o', '--out_dir', type=str, required=True)
    return parser.parse_args()


def prepare_refs(index_df: pd.DataFrame, ref_path: Path, meta_path: Path, out_dir: Path):
    fasta_path = out_dir / "references.fasta"
    tax_path = out_dir / "references.tax"

    with open(fasta_path, "w") as fasta_file, open(tax_path, "w") as tax_file:
        for accession, name, record, genus, species in ref_genomes(index_df, ref_path, meta_path):
            print(f"Handling {accession} ({name})")
            record.id = accession
            SeqIO.write([record], fasta_file, "fasta")

            tax_string = f"Bacteria;Pseudomonadota;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;{genus};{species}"
            print(
                f"{accession}\t{tax_string}",
                file=tax_file
            )
    print(f"Output reference sequences to: {fasta_path}")
    print(f"Output taxonomies to: {tax_path}")


def remove_suffixes(path: Path):
    sufs = {'.fa', '.gz', '.hdf5'}
    while path.suffix in sufs:
        path = path.with_suffix('')
    return path


def ref_genomes(index_df: pd.DataFrame, ref_path: Path, meta_path: Path) -> Iterator[Tuple[str, SeqRecord, str, str]]:
    name_to_gcf = {}
    with open(meta_path, "rt") as meta_file:
        for line in meta_file:
            tokens = line.strip().split('\t')
            name = tokens[0]
            gcf_id = tokens[2]
            name_to_gcf[name] = gcf_id

    with open(ref_path, "rt") as ref_file:
        for line in ref_file:
            rel_path = Path(line.strip())
            strain_name = remove_suffixes(rel_path).name
            gcf_id = name_to_gcf[strain_name]

            result = index_df.loc[index_df['Assembly'] == gcf_id, :].head(1)
            accession = result['Accession'].item()
            name = result['Strain'].item()
            seq_path = Path(result['SeqPath'].item())
            genus = result['Genus'].item()
            species = result['Species'].item()

            record = SeqIO.read(seq_path, "fasta")
            yield accession, name, record, genus, species


def main():
    args = parse_args()
    index_df = pd.read_csv(args.index_path, sep='\t')
    reference_file = Path(args.strainge_refs_to_keep)
    meta_file = Path(args.strainge_refs_meta)
    out_dir = Path(args.out_dir)
    prepare_refs(index_df, reference_file, meta_file, out_dir)


if __name__ == "__main__":
    main()
