import argparse
from pathlib import Path
import csv
import json

from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation
from Bio.SeqRecord import SeqRecord

from chronostrain import cfg
from chronostrain.util.entrez import fetch_genbank
from chronostrain.util.external import blastn, make_blast_db

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
    parser.add_argument('-ref', '--reference_accession', required=True, type=str,
                        help='<Required> The accession of the genome from which to extract reference gene sequences.')
    return parser.parse_args()


def get_gene_names(uniref_csv_path: Path) -> Set[str]:
    gene_names = set()
    with open(uniref_csv_path, "r") as f:
        reader = csv.reader(f, delimiter=',')

        for line_idx, row in enumerate(reader):
            if line_idx == 0:
                continue

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


def parse_records(gb_file: Path, genes_to_find: Set[str]) -> Iterator[Tuple[str, str, FeatureLocation]]:
    for record in SeqIO.parse(gb_file, format="gb"):
        for feature in record.features:
            if feature.type == "gene":
                gene_name = feature.qualifiers['gene'][0]
                locus_tag = feature.qualifiers['locus_tag'][0]

                if gene_name not in genes_to_find:
                    continue

                yield gene_name, locus_tag, feature.location


def create_chronostrain_db(reference_genes: Dict[str, Path], partial_strains: List[Dict[str, Any]], output_path: Path):
    data_dir: Path = cfg.database_cfg.data_dir
    genes_already_found: Set[str] = set()

    blast_db_dir = output_path.parent / "blast"
    blast_db_name = "Esch_coli"
    blast_db_title = "\"Escherichia coli (metaphlan markers, strainGE strains)\""
    blast_fasta_path = blast_db_dir / "genomes.fasta"
    blast_result_dir = output_path.parent / "blast_results"
    print("BLAST\n\tdatabase location: {}, \n\tresults directory: {}".format(
        str(blast_db_dir),
        str(blast_result_dir)
    ))

    # Initialize BLAST database.
    with open(blast_fasta_path, "w") as blast_fasta_file:
        for strain in partial_strains:
            marker_entries = []
            gb_file = fetch_genbank(strain['accession'], data_dir)

            genome_record = next(SeqIO.parse(gb_file, "genbank"))
            SeqIO.write(genome_record, blast_fasta_file, "fasta")

    # Run BLAST to find marker genes.
    make_blast_db(
        blast_fasta_path, blast_db_dir, blast_db_name,
        is_nucleotide=True, title=blast_db_title, parse_seqids=True
    )
    for gene_name, ref_gene_path in reference_genes.items():
        print(f"Running blastn on {gene_name}")
        blast_result_path = blast_result_dir / "gene_name.csv"
        blastn(
            db_name=blast_db_name,
            db_dir=blast_db_dir,
            query_fasta=ref_gene_path,
            evalue_max=1e-30,
            out_path=blast_result_path,
            num_threads=cfg.model_cfg.num_cores
        )

        parse_top_blast_hit(blast_result_path)

    with open(output_path, 'w') as outfile:
        json.dump(partial_strains, outfile, indent=4)

    print(f"Wrote output to {str(output_path)}.")


def parse_top_blast_hit(blast_result_path: Path):
    with open(blast_result_path, "r") as f:
        blast_result_reader = csv.reader(f)
        raise NotImplementedError("TODO")


def download_reference(accession: str, gene_names: Set[str]) -> Dict[str, Path]:
    print(f"Downloading reference accession {accession}")
    data_dir: Path = cfg.database_cfg.data_dir
    gb_file = fetch_genbank(accession, data_dir)

    genes_already_found: Set[str] = set()
    genes_to_find = set(gene_names)

    chromosome_seq = str(next(SeqIO.parse(gb_file, "gb")).seq)
    gene_paths: Dict[str, Path] = {}

    for found_gene, locus_tag, location in parse_records(gb_file, gene_names):
        print(f"Found gene {found_gene} for REF accession {accession}")

        if found_gene in genes_already_found:
            print(f"WARNING: multiple copies of {found_gene} found in {accession}. Skipping second instance.")
        else:
            genes_already_found.add(found_gene)
            genes_to_find.remove(found_gene)

            gene_out_path = data_dir / "REF_{accession}_{found_gene}.fasta"
            gene_seq = location.extract(chromosome_seq)
            print(f"Target location: {str(location)}")
            print(f"Target seq: {str(gene_seq)}")

            SeqIO.write(
                SeqRecord(gene_seq, id=f"REF_GENE_{found_gene}", description=f"{accession}_{str(location)}"),
                gene_out_path,
                "fasta"
            )
            gene_paths[found_gene] = gene_out_path

    if len(genes_to_find) > 0:
        raise RuntimeError("Couldn't find genes {}. Pick a different reference genome?".format(
            ",".join(genes_to_find))
        )

    print(f"Finished parsing reference accession {accession}.")
    return gene_paths


def main():
    args = parse_args()
    uniref_csv_path = Path(args.uniref_csv_path)
    strain_spec_path = Path(args.strain_spec_path)
    output_path = Path(args.output_path)
    strainge_db_dir = Path(args.strainge_db_dir)
    reference_accession = args.reference_accession

    gene_names = get_gene_names(uniref_csv_path)
    reference_genes = download_reference(reference_accession, gene_names)
    partial_strains = get_strain_accessions(strain_spec_path, strainge_db_dir)
    create_chronostrain_db(reference_genes, partial_strains, output_path)


if __name__ == "__main__":
    main()
