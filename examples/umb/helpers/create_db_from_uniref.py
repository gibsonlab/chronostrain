import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import json

from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation
from Bio.SeqRecord import SeqRecord

from chronostrain.util.entrez import fetch_genbank, fetch_fasta
from chronostrain.util.external import blastn, make_blast_db

from typing import List, Set, Iterator, Dict, Any, Tuple

from chronostrain.util.filesystem import convert_size
from chronostrain.util.io import read_seq_file
from chronostrain import cfg, create_logger
logger = create_logger("chronostrain.create_db_from_uniref")


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
                logger.info(f"No gene name found for Uniref entry {uniref_full}.")
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
            logger.info(f"Parsing strain information from {line}...")

            strain_name, accession = parse_strainge_path(strainge_db_dir, Path(line))
            logger.info(f"Got strain: {strain_name}, accession: {accession}")

            strain_partial_entries.append({
                'genus': 'Escherichia',
                'species': 'coli',
                'strain': strain_name,
                'accession': accession,
                'source': 'fasta',
                'markers': []
            })
    logger.info(f"Parsed {len(strain_partial_entries)} records.")
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

    blast_db_dir = output_path.parent / "blast"
    blast_db_name = "Esch_coli"
    blast_db_title = "\"Escherichia coli (metaphlan markers, strainGE strains)\""
    blast_fasta_path = blast_db_dir / "genomes.fasta"
    blast_result_dir = output_path.parent / "blast_results"
    logger.info("BLAST\n\tdatabase location: {}\n\tresults directory: {}".format(
        str(blast_db_dir),
        str(blast_result_dir)
    ))

    # Initialize BLAST database.
    blast_db_dir.mkdir(parents=True, exist_ok=True)
    strain_fasta_files = []
    for strain in partial_strains:
        accession = strain['accession']
        fasta_path = fetch_fasta(accession, data_dir)
        strain_fasta_files.append(fasta_path)

    logger.info('Concatenating {} files.'.format(len(strain_fasta_files)))
    with open(blast_fasta_path, 'w') as genome_fasta_file:
        for fpath in strain_fasta_files:
            with open(fpath, 'r') as in_file:
                for line in in_file:
                    genome_fasta_file.write(line)

    make_blast_db(
        blast_fasta_path, blast_db_dir, blast_db_name,
        is_nucleotide=True, title=blast_db_title, parse_seqids=True
    )

    # Run BLAST to find marker genes.
    blast_result_dir.mkdir(parents=True, exist_ok=True)
    for gene_name, ref_gene_path in reference_genes.items():
        gene_already_found = False
        logger.info(f"Running blastn on {gene_name}")
        blast_result_path = blast_result_dir / f"{gene_name}.tsv"
        blastn(
            db_name=blast_db_name,
            db_dir=blast_db_dir,
            query_fasta=ref_gene_path,
            evalue_max=1e-3,
            out_path=blast_result_path,
            num_threads=cfg.model_cfg.num_cores,
            out_fmt="6 saccver sstart send qstart qend sstrand evalue bitscore pident gaps qcovhsp",
            max_target_seqs=10 * len(partial_strains)  # A generous value, 10 hits per genome
        )

        locations = parse_blast_hits(blast_result_path)
        for strain_entry in partial_strains:
            for blast_hit in locations[strain_entry['accession']]:
                gene_id = f"{gene_name}_BLAST_IDX_{blast_hit.line_idx}"
                strain_entry['markers'].append(
                    {
                        'id': gene_id,
                        'name': gene_name,
                        'type': 'subseq',
                        'start': blast_hit.subj_start,
                        'end': blast_hit.subj_end,
                        'strand': blast_hit.strand,
                        'canonical': not gene_already_found
                    }
                )
                gene_already_found = True

    with open(output_path, 'w') as outfile:
        json.dump(partial_strains, outfile, indent=4)

    logger.info(f"Wrote output to {str(output_path)}.")


@dataclass
class BlastHit(object):
    line_idx: int
    subj_accession: str
    subj_start: int
    subj_end: int
    query_start: int
    query_end: int
    strand: str
    evalue: float
    bitscore: float
    pct_identity: float
    num_gaps: int
    query_coverage_per_hsp: float


def parse_blast_hits(blast_result_path: Path) -> Dict[str, List[BlastHit]]:
    accession_to_positions: Dict[str, List[BlastHit]] = defaultdict(list)
    with open(blast_result_path, "r") as f:
        blast_result_reader = csv.reader(f, delimiter='\t')
        for row_idx, row in enumerate(blast_result_reader):

            subj_acc, subj_start, subj_end, qstart, qend, sstrand, evalue, bitscore, pident, gaps, qcovhsp = row

            if sstrand == "plus":
                subj_strand = '+'
                start_pos = int(subj_start)
                end_pos = int(subj_end)
            elif sstrand == "minus":
                subj_strand = '-'
                start_pos = int(subj_end)
                end_pos = int(subj_start)
            else:
                raise RuntimeError(f"Unexpected `sstrand` token `{sstrand}`")

            accession_to_positions[subj_acc].append(
                BlastHit(
                    row_idx,
                    subj_acc,
                    start_pos,
                    end_pos,
                    int(qstart),
                    int(qend),
                    subj_strand,
                    float(evalue),
                    float(bitscore),
                    float(pident),
                    int(gaps),
                    float(qcovhsp)
                )
            )
    return accession_to_positions


def download_reference(accession: str, gene_names: Set[str]) -> Dict[str, Path]:
    logger.info(f"Downloading reference accession {accession}")
    data_dir: Path = cfg.database_cfg.data_dir
    gb_file = fetch_genbank(accession, data_dir)

    genes_already_found: Set[str] = set()
    genes_to_find = set(gene_names)

    chromosome_seq = next(SeqIO.parse(gb_file, "gb")).seq
    gene_paths: Dict[str, Path] = {}

    for found_gene, locus_tag, location in parse_records(gb_file, gene_names):
        logger.info(f"Found gene {found_gene} for REF accession {accession}")

        if found_gene in genes_already_found:
            logger.warning(f"Multiple copies of {found_gene} found in {accession}. Skipping second instance.")
        else:
            genes_already_found.add(found_gene)
            genes_to_find.remove(found_gene)

            gene_out_path = data_dir / f"REF_{accession}_{found_gene}.fasta"
            gene_seq = location.extract(chromosome_seq)
            SeqIO.write(
                SeqRecord(gene_seq, id=f"REF_GENE_{found_gene}", description=f"{accession}_{str(location)}"),
                gene_out_path,
                "fasta"
            )
            gene_paths[found_gene] = gene_out_path

    if len(genes_to_find) > 0:
        logger.info("Couldn't find genes {}.".format(
            ",".join(genes_to_find))
        )

    logger.info(f"Finished parsing reference accession {accession}.")
    return gene_paths


def print_summary(strain_entries: List[Dict[str, Any]], genes: Set[str]):
    for strain_entry in strain_entries:
        accession = strain_entry['accession']
        found_genes = set()
        for marker_entry in strain_entry['markers']:
            found_genes.add(marker_entry['name'])

        genes_not_found = genes.difference(found_genes)
        if len(genes_not_found) > 0:
            logger.info("Accession {}, {} genes not found: [{}]".format(
                accession,
                len(genes_not_found),
                ', '.join(genes_not_found)
            ))


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
    print_summary(partial_strains, gene_names)


if __name__ == "__main__":
    main()
