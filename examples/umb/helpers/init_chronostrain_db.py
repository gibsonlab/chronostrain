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
import pandas as pd

from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation
from Bio.SeqRecord import SeqRecord
from bioservices import UniProt

from chronostrain.util.entrez import fetch_genbank
from chronostrain.util.external import make_blast_db, blastn

from typing import List, Set, Dict, Any, Tuple, Iterator

from chronostrain.util.io import read_seq_file
from chronostrain import cfg, create_logger
logger = create_logger("chronostrain.init_db")


READ_LEN = 150


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create chronostrain database file from specified gene_info_uniref marker CSV."
    )

    # Input specification.
    parser.add_argument('-m', '--metaphlan_pkl_path', required=True, type=str,
                        help='<Required> The path to the metaphlan pickle database file.')
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='<Required> The path to the target output chronostrain db json file.')
    parser.add_argument('-r', '--refseq_dir', required=True, type=str,
                        help='<Required> The strainGE database directory.')

    # Optional params
    parser.add_argument('--reference_accession', required=False, type=str,
                        default='U00096.3',
                        help='<Optional> The reference genome to use for pulling out annotated gene sequences.')
    return parser.parse_args()


# ============================= Common resources
def strain_seq_dir(strain_id: str) -> Path:
    return cfg.database_cfg.data_dir / "assemblies" / strain_id


# ============================= Creation of database configuration (Indexing + partial JSON creation)
def perform_indexing(refseq_dir: Path) -> pd.DataFrame:
    df_entries = []

    if not refseq_dir.is_dir():
        raise RuntimeError(f"Provided path `{refseq_dir}` is not a directory.")
    for genus_dir in (refseq_dir / "human_readable" / "refseq" / "bacteria").iterdir():
        if not genus_dir.is_dir():
            continue

        genus = genus_dir.name
        for species_dir in genus_dir.iterdir():
            if not species_dir.is_dir():
                continue

            species = species_dir.name
            logger.info(f"Searching through {genus} {species}...")

            for strain_dir in species_dir.iterdir():
                if not strain_dir.is_dir():
                    continue

                strain_name = strain_dir.name
                target_files = list(strain_dir.glob("*_genomic.fna.gz"))

                for fpath in target_files:
                    if fpath.name.endswith('_cds_from_genomic.fna.gz'):
                        continue

                    if fpath.name.endswith('_rna_from_genomic.fna.gz'):
                        continue

                    for accession, chrom_path in extract_chromosomes(fpath):
                        logger.debug("Found accession {} ({} {}, Strain `{}`)".format(
                            accession,
                            genus,
                            species,
                            strain_name
                        ))
                        df_entries.append({
                            "Genus": genus,
                            "Species": species,
                            "Strain": strain_name,
                            "Accession": accession,
                            "SeqPath": chrom_path
                        })
    return pd.DataFrame(df_entries)


def extract_chromosomes(path: Path) -> Iterator[Tuple[str, Path]]:
    for record in read_seq_file(path, file_format='fasta'):
        desc = record.description
        accession = record.id.split(' ')[0]

        if "plasmid" in desc or len(record.seq) < 500000:
            continue
        else:
            chrom_path = path.parent / f"{accession}.chrom.fna"
            SeqIO.write([record], chrom_path, "fasta")

            yield accession, chrom_path


# ============= Rest of initialization (Run BLAST)
def create_chronostrain_db(
        gene_paths: Dict[str, Path],
        seq_index: pd.DataFrame,
        output_path: Path
) -> List[Dict[str, Any]]:
    """
    :param gene_paths:
    :param seq_index: A DataFrame with five columns: "Genus", "Species", "Strain", "Accession", "SeqPath".
    :param output_path:
    :return:
    """
    # ======== BLAST configuration
    blast_db_dir = output_path.parent / "blast"
    blast_db_name = "db_esch"
    blast_db_title = "\"Escherichia (metaphlan markers, NCBI complete)\""
    blast_fasta_path = blast_db_dir / "genomes.fasta"
    blast_result_dir = output_path.parent / "blast_results"
    logger.info("BLAST\n\tdatabase location: {}\n\tresults directory: {}".format(
        str(blast_db_dir),
        str(blast_result_dir)
    ))

    # ========= Initialize BLAST database.
    blast_db_dir.mkdir(parents=True, exist_ok=True)
    strain_fasta_files = []
    strain_entries = []
    for idx, row in seq_index.iterrows():
        accession = row['Accession']
        strain_id = accession  # Use the accession as the strain's ID (we are only using full chromosome assemblies).

        fasta_path = row['SeqPath']
        strain_fasta_files.append(fasta_path)
        strain_entries.append({
            'id': strain_id,
            'genus': row['Genus'],
            'species': row['Species'],
            'name': row['Strain'],
            'seqs': [{'accession': accession, 'seq_type': 'chromosome'}],
            'markers': []
        })

        symlink_path = strain_seq_dir(strain_id) / f"{accession}.fasta"
        if symlink_path.exists():
            logger.info(f"Path {symlink_path} already exists.")
        else:
            symlink_path.parent.mkdir(exist_ok=True, parents=True)
            symlink_path.symlink_to(fasta_path)
            logger.info(f"Symlink {symlink_path} -> {fasta_path}")

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

    Path(blast_fasta_path).unlink()  # clean up large fasta file.

    # Run BLAST to find marker genes.
    blast_result_dir.mkdir(parents=True, exist_ok=True)
    for gene_name, ref_gene_path in gene_paths.items():
        gene_already_found = False
        logger.info(f"Running blastn on {gene_name}.")
        blast_result_path = blast_result_dir / f"{gene_name}.tsv"
        blastn(
            db_name=blast_db_name,
            db_dir=blast_db_dir,
            query_fasta=ref_gene_path,
            evalue_max=1e-3,
            out_path=blast_result_path,
            num_threads=cfg.model_cfg.num_cores,
            out_fmt="6 saccver sstart send qstart qend evalue bitscore pident gaps qcovhsp",
            max_target_seqs=10 * seq_index.shape[0],  # A generous value, 10 hits per genome
            strand="both"
        )

        logger.debug(f"Parsing BLAST hits for gene `{gene_name}`.")
        locations = parse_blast_hits(blast_result_path)
        for strain_entry in strain_entries:
            seq_accession = strain_entry['seqs'][0]['accession']
            for blast_hit in locations[seq_accession]:
                gene_id = f"{gene_name}_BLASTIDX_{blast_hit.line_idx}"
                strain_entry['markers'].append(
                    {
                        'id': gene_id,
                        'name': gene_name,
                        'type': 'subseq',
                        'source': seq_accession,
                        'start': blast_hit.subj_start,
                        'end': blast_hit.subj_end,
                        'strand': blast_hit.strand,
                        'canonical': not gene_already_found
                    }
                )
                gene_already_found = True

    return prune_entries(strain_entries)


def prune_entries(strain_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for strain_entry in strain_entries:
        logger.info("No markers found for "
                    f"{strain_entry['genus']} {strain_entry['species']} "
                    f"{strain_entry['name']} "
                    f"(ID {strain_entry['id']}).")
    return [
        strain_entry
        for strain_entry in strain_entries
        if len(strain_entry['markers']) > 0
    ]


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


# ======================== Reference genome: pull from downloaded genbank file.
def download_reference(accession: str, metaphlan_pkl_path: Path) -> Dict[str, Path]:
    logger.info(f"Downloading reference accession {accession}")
    target_dir = cfg.database_cfg.data_dir / "reference" / accession
    target_dir.mkdir(exist_ok=True, parents=True)

    gb_file = fetch_genbank(accession, target_dir)

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

            gene_out_path = target_dir / f"{found_cluster}.fasta"
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
            logger.debug(
                f"No result found for UniProt query `{uniprot_id}`, derived from metaphlan ID `{metaphlan_marker_id}`."
            )
            continue
        else:
            logger.debug(f"Found {len(res)} hits for UniProt query `{uniprot_id}`.")

        gene_name = res[uniprot_id]['Entry name']
        cluster = res[uniprot_id]['Gene names'].split()
        yield gene_name, cluster


def parse_genbank_genes(gb_file: Path) -> Iterator[Tuple[str, str, FeatureLocation]]:
    for record in SeqIO.parse(gb_file, format="gb"):
        for feature in record.features:
            if feature.type == "gene":
                gene_name = feature.qualifiers['gene'][0]
                locus_tag = feature.qualifiers['locus_tag'][0]

                yield gene_name, locus_tag, feature.location


# ================= MAIN
def print_summary(strain_entries: List[Dict[str, Any]], gene_paths: Dict[str, Path]):
    for strain_entry in strain_entries:
        seq_accession = strain_entry['seqs'][0]['accession']
        found_genes: Set[str] = set()
        for marker_entry in strain_entry['markers']:
            found_genes.add(marker_entry['name'])

        genes_not_found = set(gene_paths.keys()).difference(found_genes)
        if len(genes_not_found) > 0:
            logger.info("Accession {}, {} genes not found: [{}]".format(
                seq_accession,
                len(genes_not_found),
                ', '.join(genes_not_found)
            ))


def main():
    args = parse_args()
    metaphlan_pkl_path = Path(args.metaphlan_pkl_path)
    output_path = Path(args.output_path)
    seq_dir = Path(args.refseq_dir)

    # ================= Indexing of refseqs
    logger.info(f"Indexing refseqs located in {seq_dir}")
    seq_index = perform_indexing(seq_dir)
    index_path = seq_dir / "index.tsv"
    seq_index.to_csv(index_path, sep='\t')
    logger.info(f"Wrote index to {str(index_path)}.")

    # ================= Pull out reference genes
    logger.info(f"Retrieving reference genes from {args.reference_accession}")
    ref_gene_paths = download_reference(args.reference_accession, metaphlan_pkl_path)

    # ================= Compile into JSON.
    logger.info("Creating JSON entries.")
    object_entries = create_chronostrain_db(ref_gene_paths, seq_index, output_path)

    print_summary(object_entries, ref_gene_paths)
    with open(output_path, 'w') as outfile:
        json.dump(object_entries, outfile, indent=4)
    logger.info(f"Wrote output to {str(output_path)}.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.error(e)
        raise
