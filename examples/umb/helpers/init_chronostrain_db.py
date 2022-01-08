"""
Script which creates a chronostrain db of all UNIREF-specified marker genes.
Works in 3 steps:
    1) For each marker gene name, extract its annotation from reference K-12 genome.
    2) Run BLAST to find all matching hits (even partial hits) of marker gene sequence to each strain assembly
        (with blastn configured to allow, on average, up to 10 hits per genome).
    3) Convert BLAST results into chronostrain JSON database specification.
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import pickle
import bz2

from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation
from Bio.SeqRecord import SeqRecord
from bioservices import UniProt

from chronostrain.util.entrez import fetch_fasta, fetch_genbank
from chronostrain.util.external import tblastn, make_blast_db

from typing import List, Set, Dict, Any, Tuple, Iterator

from chronostrain.util.io import read_seq_file
from chronostrain import cfg, create_logger
logger = create_logger("chronostrain.create_db_from_uniref")


READ_LEN = 150


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create chronostrain database file from specified gene_info_uniref marker CSV."
    )

    # Input specification.
    parser.add_argument('-m', '--metaphlan_pkl_path', required=True, type=str,
                        help='<Required> The path to the metaphlan pickle database file.')
    parser.add_argument('-s', '--strain_spec_path', required=True, type=str,
                        help='<Required> The path listing strain genome files '
                             '(references_to_keep.txt from StrainGE tutorial).')
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='<Required> The path to the target output chronostrain db json file.')
    parser.add_argument('-sdb', '--strainge_db_dir', required=True, type=str,
                        help='<Required> The strainGE database directory.')
    return parser.parse_args()


def parse_genbank_genes(gb_file: Path) -> Iterator[Tuple[str, str, FeatureLocation]]:
    for record in SeqIO.parse(gb_file, format="gb"):
        for feature in record.features:
            if feature.type == "gene":
                gene_name = feature.qualifiers['gene'][0]
                locus_tag = feature.qualifiers['locus_tag'][0]

                yield gene_name, locus_tag, feature.location


def metaphlan_markers(metaphlan_db: Dict, metaphlan_clade: str):
    for marker_key, marker_entry in metaphlan_db['markers'].items():
        if metaphlan_clade in marker_entry['taxon']:
            yield marker_key


def get_marker_genes(metaphlan_pkl_path: Path) -> Iterator[Tuple[str, List[str]]]:
    db = pickle.load(bz2.open(metaphlan_pkl_path, 'r'))
    u = UniProt()

    # for metaphlan_marker_id in metaphlan_markers(db, 'g__Escherichia'):
    for metaphlan_marker_id in metaphlan_markers(db, 's__Escherichia_coli'):
        tokens = metaphlan_marker_id.split('__')
        uniprot_id = tokens[1]

        res = u.quick_search(uniprot_id)

        if uniprot_id not in res:
            print(
                f"No result found for UniProt query `{uniprot_id}`, derived from metaphlan ID `{metaphlan_marker_id}`."
            )
            continue
        else:
            print(f"Found {len(res)} hits for UniProt query `{uniprot_id}`.")

        gene_name = res[uniprot_id]['Entry name']
        cluster = res[uniprot_id]['Gene names'].split()
        yield gene_name, cluster


def parse_strainge_path(strainge_db_dir: Path, hdf5_path: Path) -> Tuple[str, str, str, str]:
    fa_filename = hdf5_path.with_suffix("").name
    fa_path = strainge_db_dir / fa_filename

    full_suffix = '.fa.gz.hdf5'
    base_tokens = hdf5_path.name[:-len(full_suffix)].split("_")
    strain_name = "_".join(base_tokens[2:])
    genus_name = base_tokens[0]
    species_name = base_tokens[1]

    for record in read_seq_file(fa_path, "fasta"):
        accession = record.id.strip().split(" ")[0]
        return genus_name, species_name, strain_name, accession

    raise RuntimeError(f"Couldn't find a valid record in {str(fa_path)}.")


def get_strain_accessions(strain_spec_path: Path, strainge_db_dir: Path) -> List[Dict[str, Any]]:
    strain_partial_entries = []
    with open(strain_spec_path, "r") as strain_file:
        for line in strain_file:
            line = line.strip()
            if len(line) == 0:
                continue
            logger.info(f"Parsing strain information from {line}...")

            genus, species, strain_name, accession = parse_strainge_path(strainge_db_dir, Path(line))
            logger.info(f"Got strain: {strain_name}, accession: {accession}")

            strain_partial_entries.append({
                'genus': genus,
                'species': species,
                'strain': strain_name,
                'accession': accession,
                'source': 'fasta',
                'markers': []
            })
    logger.info(f"Parsed {len(strain_partial_entries)} records.")
    return strain_partial_entries


def create_chronostrain_db(gene_paths: Dict[str, Path], partial_strains: List[Dict[str, Any]], output_path: Path):
    data_dir: Path = cfg.database_cfg.data_dir

    blast_db_dir = output_path.parent / "blast"
    blast_db_name = "strainge_db"
    blast_db_title = "\"Escherichia (metaphlan markers, strainGE strains)\""
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
    for gene_name, ref_gene_path in gene_paths.items():
        gene_already_found = False
        logger.info(f"Running blastn on {gene_name}.")
        blast_result_path = blast_result_dir / f"{gene_name}.tsv"
        tblastn(
            db_name=blast_db_name,
            db_dir=blast_db_dir,
            query_fasta=ref_gene_path,
            evalue_max=1e-3,
            out_path=blast_result_path,
            num_threads=cfg.model_cfg.num_cores,
            out_fmt="6 saccver sstart send qstart qend evalue bitscore pident gaps qcovhsp",
            max_target_seqs=10 * len(partial_strains)  # A generous value, 10 hits per genome
        )

        print(f"Parsing BLAST hits for gene `{gene_name}`.")
        locations = parse_blast_hits(blast_result_path)
        for strain_entry in partial_strains:
            for blast_hit in locations[strain_entry['accession']]:
                gene_id = f"{gene_name}_BLASTIDX_{blast_hit.line_idx}"
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

    strain_entries = [
        strain_entry
        for strain_entry in partial_strains
        if len(strain_entry['markers']) > 0
    ]

    with open(output_path, 'w') as outfile:
        json.dump(strain_entries, outfile, indent=4)

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

            subj_acc, subj_start, subj_end, qstart, qend, evalue, bitscore, pident, gaps, qcovhsp = row

            subj_start = int(subj_start)
            subj_end = int(subj_end)

            if subj_start < subj_end:
                start_pos = subj_start
                end_pos = subj_end
                strand = '+'
            else:
                start_pos = subj_end
                end_pos = subj_start
                strand = '-'

            if end_pos - start_pos + 1 < READ_LEN:
                continue

            accession_to_positions[subj_acc].append(
                BlastHit(
                    row_idx,
                    subj_acc,
                    start_pos,
                    end_pos,
                    int(qstart),
                    int(qend),
                    strand,
                    float(evalue),
                    float(bitscore),
                    float(pident),
                    int(gaps),
                    float(qcovhsp)
                )
            )
    return accession_to_positions


def download_reference(accession: str, metaphlan_pkl_path: Path) -> Dict[str, Path]:
    logger.info(f"Downloading reference accession {accession}")
    data_dir: Path = cfg.database_cfg.data_dir
    gb_file = fetch_genbank(accession, data_dir)

    clusters_already_found: Set[str] = set()
    clusters_to_find: Set[str] = set()
    genes_to_clusters: Dict[str, str] = {}

    for cluster, cluster_genes in get_marker_genes(metaphlan_pkl_path):
        clusters_to_find.add(cluster)
        for gene in cluster_genes:
            genes_to_clusters[gene.lower()] = cluster

    chromosome_seq = next(SeqIO.parse(gb_file, "gb")).seq
    gene_paths: Dict[str, Path] = {}

    for gb_gene_name, locus_tag, location in parse_genbank_genes(gb_file):
        if gb_gene_name.lower() not in genes_to_clusters:
            continue
        if genes_to_clusters[gb_gene_name.lower()] not in clusters_to_find:
            continue

        found_cluster = genes_to_clusters[gb_gene_name.lower()]
        logger.info(f"Found uniref cluster {found_cluster} for accession {accession} (name={gb_gene_name})")

        if found_cluster in clusters_already_found:
            logger.warning(f"Multiple copies of {found_cluster} found in {accession}. Skipping second instance.")
        else:
            clusters_already_found.add(found_cluster)
            clusters_to_find.remove(found_cluster)

            gene_out_path = data_dir / f"REF_{accession}_{found_cluster}.fasta"
            gene_seq = location.extract(chromosome_seq)
            SeqIO.write(
                SeqRecord(gene_seq, id=f"REF_GENE_{found_cluster}", description=f"{accession}_{str(location)}"),
                gene_out_path,
                "fasta"
            )
            gene_paths[found_cluster] = gene_out_path

    if len(clusters_to_find) > 0:
        logger.info("Couldn't find genes {}.".format(
            ",".join(clusters_to_find))
        )

    logger.info(f"Finished parsing reference accession {accession}.")
    return gene_paths


def print_summary(strain_entries: List[Dict[str, Any]], gene_paths: Dict[str, Path]):
    for strain_entry in strain_entries:
        accession = strain_entry['accession']
        found_genes: Set[str] = set()
        for marker_entry in strain_entry['markers']:
            found_genes.add(marker_entry['name'])

        genes_not_found = set(gene_paths.keys()).difference(found_genes)
        if len(genes_not_found) > 0:
            logger.info("Accession {}, {} genes not found: [{}]".format(
                accession,
                len(genes_not_found),
                ', '.join(genes_not_found)
            ))


def main():
    args = parse_args()
    metaphlan_pkl_path = Path(args.metaphlan_pkl_path)
    strain_spec_path = Path(args.strain_spec_path)
    output_path = Path(args.output_path)
    strainge_db_dir = Path(args.strainge_db_dir)

    gene_paths = download_reference("NC_000913.3", metaphlan_pkl_path)
    partial_strains = get_strain_accessions(strain_spec_path, strainge_db_dir)
    create_chronostrain_db(gene_paths, partial_strains, output_path)
    print_summary(partial_strains, gene_paths)


if __name__ == "__main__":
    main()
