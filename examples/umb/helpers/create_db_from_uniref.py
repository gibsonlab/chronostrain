import argparse
from pathlib import Path
import csv
import json

from Bio import SeqIO
from chronostrain import cfg
from chronostrain.util.entrez import fetch_genbank

from typing import List, Set, Iterator, Dict, Any, Tuple

from chronostrain.util.io import read_seq_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create chronostrain database file from specified gene_info_uniref marker CSV."
    )

    # Input specification.
    parser.add_argument('-u', '--uniref_csv_path', required=True, type=str,
                        help='<Required> The path to the CSV input file.')
    parser.add_argument('-s', '--strain_spec_path', required=True, type=str,
                        help='<Required> The path to the CSV file of strain accessions.')
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='<Required> The path to the target output chronostrain db json file.')
    parser.add_argument('-sdb', '--strainge_db_dir', required=True, type=str,
                        help='<Required> The strainGE database directory.')
    return parser.parse_args()


def get_gene_names(uniref_csv_path: Path) -> Set[str]:
    gene_names = set()
    with open(uniref_csv_path, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            uniref_full, uniref_id, gene_name = row
            if len(gene_name.strip()) == 0:
                print(f"No gene name found for Uniref entry {uniref_full}.")
                continue

            gene_names.add(gene_name)
    return gene_names


def parse_strainge_path(strainge_db_dir: Path, hdf5_path: Path) -> Tuple[str, str]:
    fa_filename = hdf5_path.with_suffix("").name
    fa_path = strainge_db_dir / fa_filename

    suffix = '.fa.gz.hdf5'
    base_tokens = hdf5_path.name[:-len(suffix)].split("_")
    strain_name = "_".join(base_tokens[2:])

    for record in read_seq_file(fa_path, "fasta"):
        accession = record.id.strip().split(" ")[0]
        return strain_name, accession

    raise RuntimeError(f"Couldn't find a valid record in {str(fa_path)}.")


def get_strain_accessions(strain_spec_path: Path, strainge_db_dir: Path) -> List[Dict[str, Any]]:
    strain_partial_entries = []
    with open(strain_spec_path, "r") as strain_file:
        for line in strain_file:
            line = line.strip()
            if len(line) == 0:
                continue
            print(f"Parsing strain information from {line}...")

            strain_name, accession = parse_strainge_path(strainge_db_dir, Path(line))
            print(f"Got strain: {strain_name}, accession: {accession}")

            strain_partial_entries.append({
                'genus': 'Escherichia',
                'species': 'coli',
                'strain': strain_name,
                'accession': accession,
                'source': 'genbank'
            })
    return strain_partial_entries


def parse_records(gb_file: Path, genes_to_find: Set[str]) -> Iterator[Tuple[str, str]]:
    for record in SeqIO.parse(gb_file, format="gb"):
        for feature in record.features:
            if feature.type == "gene":
                gene_name = feature.qualifiers['gene'][0]
                locus_tag = feature.qualifiers['locus_tag'][0]

                if gene_name not in genes_to_find:
                    continue

                yield gene_name, locus_tag


def create_chronostrain_db(gene_names: Set[str], partial_strains: List[Dict[str, Any]], output_path: Path):
    data_dir: Path = cfg.database_cfg.data_dir
    genes_already_found: Set[str] = set()

    for strain in partial_strains:
        marker_entries = []
        gb_file = fetch_genbank(strain['accession'], data_dir)
        genes_to_find = set(gene_names)

        for found_gene, locus_tag in parse_records(gb_file, gene_names):
            print(f"Found gene {found_gene} for accession {strain['accession']}")

            if found_gene in genes_already_found:
                is_canonical = False
            else:
                is_canonical = True
                genes_already_found.add(found_gene)

            genes_to_find.remove(found_gene)
            marker_entries.append({
                'type': 'tag',
                'locus_tag': locus_tag,
                'name': found_gene,
                'canonical': is_canonical
            })

        print("Couldn't find genes {}".format(",".join(genes_to_find)))
        strain['markers'] = marker_entries

    with open(output_path, 'w') as outfile:
        json.dump(partial_strains, outfile, indent=4)

    print(f"Wrote output to {str(output_path)}.")


def main():
    args = parse_args()
    uniref_csv_path = Path(args.uniref_csv_path)
    strain_spec_path = Path(args.strain_spec_path)
    output_path = Path(args.output_path)
    strainge_db_dir = Path(args.strainge_db_dir)

    gene_names = get_gene_names(uniref_csv_path)
    partial_strains = get_strain_accessions(strain_spec_path, strainge_db_dir)
    create_chronostrain_db(gene_names, partial_strains, output_path)


if __name__ == "__main__":
    main()
