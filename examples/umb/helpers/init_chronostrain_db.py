"""
Script which creates a chronostrain db of all UNIREF-specified marker genes.
Works in 3 steps:
    1) For each marker gene name, extract its annotation from reference K-12 genome.
    2) Run BLAST to find all matching hits (even partial hits) of marker gene sequence to each strain assembly
        (with blastn configured to allow, on average, up to 10 hits per genome).
    3) Convert BLAST results into chronostrain JSON database specification.
"""
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
import csv
import json
import pickle
import bz2

import pandas as pd

from Bio import SeqIO, Entrez
from Bio.SeqFeature import FeatureLocation
from Bio.SeqRecord import SeqRecord
from bioservices import UniProt

from chronostrain.util.entrez import fetch_genbank
from chronostrain.util.external import make_blast_db, blastn

from typing import List, Set, Dict, Any, Tuple, Iterator, Optional

from chronostrain.util.io import read_seq_file
from chronostrain import cfg, create_logger
logger = create_logger("chronostrain.init_db")


READ_LEN = 150
Entrez.email = cfg.entrez_cfg.email


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create chronostrain database file from specified gene_info_uniref marker CSV."
    )

    # Input specification.
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='<Required> The path to the target output chronostrain db json file.')
    parser.add_argument('-dbdir', '--blast_db_dir', required=True, type=str,
                        help='<Required> The path to the blast database directory.')
    parser.add_argument('-dbname', '--blast_db_name', required=True, type=str,
                        help='<Required> The name of the BLAST database.')
    parser.add_argument('-r', '--refseq_index', required=True, type=str,
                        help='<Required> Path to the RefSeq index TSV file.')

    # Optional params
    parser.add_argument('--min_pct_idty', required=False, type=int, default=75,
                        help='<Optional> The percent identity threshold for BLAST. (default: 75)')
    parser.add_argument('--max_target_seqs', required=False, type=int, default=100000,
                        help='<Optional> The max # of alignments to output for each BLAST query. (default: 100000)')
    parser.add_argument('--metaphlan_pkl_path', required=False, type=str,
                        help='<Optional> The path to the metaphlan pickle database file.')
    parser.add_argument('--reference_accession', required=False, type=str,
                        default='U00096.3',
                        help='<Optional> The reference genome to use for pulling out annotated gene sequences.')
    parser.add_argument('--uniprot_csv', required=False, type=str,
                        default='',
                        help='<Optional> A path to a two-column CSV file (<UniprotID>, <ClusterName>) format specifying'
                             'any desired additional genes not given by metaphlan.')
    return parser.parse_args()


# ===================================================== BEGIN Local resources

def create_blast_db(blast_db_dir: Path,
                    blast_db_name: str,
                    blast_db_title: str,
                    strain_fasta_files: List[Path]):
    # ========= Initialize BLAST database.
    blast_fasta_path = blast_db_dir / "genomes.fasta"
    blast_db_dir.mkdir(parents=True, exist_ok=True)
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


def run_blast_local(db_dir: Path,
                    db_name: str,
                    result_dir: Path,
                    gene_paths: Dict[str, Path],
                    max_target_seqs: int,
                    min_pct_idty: int,
                    out_fmt: str) -> Dict[str, Path]:
    # Retrieve taxIDs for bacteria.
    logger.info("Retrieving taxonomic IDS for bacteria.")

    # Run BLAST.
    result_paths: Dict[str, Path] = {}
    result_dir.mkdir(parents=True, exist_ok=True)
    for gene_name, ref_gene_path in gene_paths.items():
        logger.info(f"Running blastn on {gene_name}.")
        blast_result_path = result_dir / f"{gene_name}.tsv"
        blastn(
            db_name=db_name,
            db_dir=db_dir,
            query_fasta=ref_gene_path,
            perc_identity_cutoff=min_pct_idty,
            out_path=blast_result_path,
            num_threads=cfg.model_cfg.num_cores,
            out_fmt=out_fmt,
            max_target_seqs=max_target_seqs,
            strand="both"
        )
        result_paths[gene_name] = blast_result_path
    return result_paths
# ===================================================== END Local resources


# ============= Rest of initialization
def create_chronostrain_db(
        blast_result_dir: Path,
        strain_df: pd.DataFrame,
        gene_paths: Dict[str, Path],
        blast_db_dir: Path,
        blast_db_name: str,
        min_pct_idty: int,
        max_target_seqs: int
) -> List[Dict[str, Any]]:
    """
    :return:
    """

    blast_results = run_blast_local(
        db_dir=blast_db_dir,
        db_name=blast_db_name,
        result_dir=blast_result_dir,
        gene_paths=gene_paths,
        max_target_seqs=max_target_seqs,
        min_pct_idty=min_pct_idty,
        out_fmt=_BLAST_OUT_FMT
    )

    return create_strain_entries(blast_results, gene_paths, strain_df)


def blank_strain_entry(strain_id: str, genus: str, species: str, name: str, accession: str):
    return {
        'id': strain_id,
        'genus': genus,
        'species': species,
        'name': name,
        'seqs': [{'accession': accession, 'seq_type': 'chromosome'}],
        'markers': []
    }


def create_strain_entries(blast_results: Dict[str, Path], ref_gene_paths: Dict[str, Path], strain_df: pd.DataFrame):
    strain_entries: Dict[str, Dict] = {}
    seen_accessions: Set[str] = set()

    for gene_name, blast_result_path in blast_results.items():
        blast_hits = parse_blast_hits(blast_result_path)

        # Parse the entries.
        canonical_gene_found = False
        ref_gene_path = ref_gene_paths[gene_name]
        ref_gene_len = len(next(iter(read_seq_file(ref_gene_path, 'fasta'))).seq)
        min_canonical_length = ref_gene_len * 0.95

        logger.debug(f"Parsing BLAST hits for gene `{gene_name}`.")
        for subj_acc in blast_hits.keys():
            # Create strain entries if they don't already exist.
            if subj_acc not in seen_accessions:
                seen_accessions.add(subj_acc)
                strain_row = strain_df.loc[strain_df['Accession'] == subj_acc, ['Genus', 'Species', 'Strain']].head(1)
                subj_genus = strain_row['Genus'].item()
                subj_species = strain_row['Species'].item()
                subj_strain_name = strain_row['Strain'].item()

                strain_entries[subj_acc] = blank_strain_entry(
                    strain_id=subj_acc,
                    genus=subj_genus,
                    species=subj_species,
                    name=subj_strain_name,
                    accession=subj_acc
                )

            if subj_acc not in strain_entries:
                continue

            strain_entry = strain_entries[subj_acc]
            seq_accession = strain_entry['seqs'][0]['accession']
            for blast_hit in blast_hits[seq_accession]:
                is_canonical = (
                        (blast_hit.subj_end - blast_hit.subj_start) >= min_canonical_length
                        and
                        not canonical_gene_found
                )

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
                        'canonical': is_canonical
                    }
                )
                canonical_gene_found = canonical_gene_found or is_canonical

    return prune_entries([entry for _, entry in strain_entries.items()])


def prune_entries(strain_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for strain_entry in strain_entries:
        if len(strain_entry['markers']) == 0:
            logger.info("No markers found for "
                        f"{strain_entry['genus']} {strain_entry['species']} "
                        f"{strain_entry['name']} "
                        f"(ID {strain_entry['id']}).")
    return [
        strain_entry
        for strain_entry in strain_entries
        if len(strain_entry['markers']) > 0
    ]


_BLAST_OUT_FMT = "6 saccver sstart send slen qstart qend evalue pident gaps qcovhsp"


class BlastHit(object):
    def __init__(self,
                 line_idx: int,
                 subj_accession: str,
                 subj_start: int,
                 subj_end: int,
                 subj_len: int,
                 query_start: int,
                 query_end: int,
                 strand: str,
                 evalue: float,
                 pct_identity: float,
                 num_gaps: int,
                 query_coverage_per_hsp: float,
                 ):
        self.line_idx = line_idx
        self.subj_accession = subj_accession
        self.subj_start = subj_start
        self.subj_end = subj_end
        self.subj_len = subj_len
        self.query_start = query_start
        self.query_end = query_end
        self.strand = strand
        self.evalue = evalue
        self.pct_identity = pct_identity
        self.num_gaps = num_gaps
        self.query_coverage_per_hsp = query_coverage_per_hsp


def parse_blast_hits(blast_result_path: Path) -> Dict[str, List[BlastHit]]:
    accession_to_positions: Dict[str, List[BlastHit]] = defaultdict(list)
    with open(blast_result_path, "r") as f:
        blast_result_reader = csv.reader(f, delimiter='\t')
        for row_idx, row in enumerate(blast_result_reader):
            subj_acc, subj_start, subj_end, subj_len, qstart, qend, evalue, pident, gaps, qcovhsp = row

            subj_start = int(subj_start)
            subj_end = int(subj_end)
            subj_len = int(subj_len)

            if subj_start < subj_end:
                subj_start_pos = subj_start
                subj_end_pos = subj_end
                strand = '+'
            else:
                subj_start_pos = subj_end
                subj_end_pos = subj_start
                strand = '-'

            if subj_end_pos - subj_start_pos + 1 < READ_LEN:
                continue

            accession_to_positions[subj_acc].append(
                BlastHit(
                    line_idx=row_idx,
                    subj_accession=subj_acc,
                    subj_start=subj_start_pos,
                    subj_end=subj_end_pos,
                    subj_len=subj_len,
                    query_start=int(qstart),
                    query_end=int(qend),
                    strand=strand,
                    evalue=float(evalue),
                    pct_identity=float(pident),
                    num_gaps=int(gaps),
                    query_coverage_per_hsp=float(qcovhsp)
                )
            )
    return accession_to_positions


# ======================== Reference genome: pull from downloaded genbank file.
def download_reference(accession: str, metaphlan_pkl_path: Path, uniprot_csv_path: Path) -> Dict[str, Path]:
    logger.info(f"Downloading reference accession {accession}")
    target_dir = cfg.database_cfg.data_dir / "reference" / accession
    target_dir.mkdir(exist_ok=True, parents=True)

    gb_file = fetch_genbank(accession, target_dir)

    clusters_already_found: Set[str] = set()
    clusters_to_find: Set[str] = set()
    genes_to_clusters: Dict[str, str] = {}

    for cluster, cluster_genes in get_marker_genes(metaphlan_pkl_path, uniprot_csv_path):
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


def metaphlan_markers(metaphlan_pkl_path: Path, metaphlan_clade: str) -> Iterator[str]:
    if metaphlan_pkl_path is None:
        yield from []

    db = pickle.load(bz2.open(metaphlan_pkl_path, 'r'))
    for marker_key, marker_entry in db['markers'].items():
        if metaphlan_clade in marker_entry['taxon']:
            tokens = marker_key.split('__')
            uniprot_id = tokens[1]
            logger.debug(
                f"Found metaphlan marker key `{marker_key}`, parsed UniProt ID `{uniprot_id}`."
            )
            yield uniprot_id


def get_marker_genes(
        metaphlan_pkl_path: Optional[Path],
        uniprot_csv_path: Optional[Path]
) -> Iterator[Tuple[str, List[str]]]:
    u = UniProt()

    sources = []
    if metaphlan_pkl_path is not None:
        sources.append(metaphlan_markers(metaphlan_pkl_path, 's__Escherichia_coli'))
    if uniprot_csv_path is not None:
        sources.append(parse_uniprot_csv(uniprot_csv_path))

    # for metaphlan_marker_id in metaphlan_markers(db, 'g__Escherichia'):
    for uniprot_id in itertools.chain(*sources):
        res = u.quick_search(uniprot_id)

        if uniprot_id not in res:
            logger.debug(
                f"No result found for UniProt query `{uniprot_id}`."
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


def parse_uniprot_csv(csv_path: Path) -> Iterator[str]:
    with open(csv_path, "r") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue

            tokens = line.strip().split(",")
            uniprot_id, cluster_name = tokens[0], tokens[1]

            if uniprot_id == "UNIPROT_ID":
                # Header line
                continue

            logger.debug(f"Searching additional cluster `{cluster_name}`, uniprot ID `{uniprot_id}`")
            yield uniprot_id


def main():
    args = parse_args()
    output_path = Path(args.output_path)

    # ================= Pull out reference genes
    logger.info(f"Retrieving reference genes from {args.reference_accession}")
    ref_gene_paths = download_reference(args.reference_accession, args.metaphlan_pkl_path, args.uniprot_csv)

    refseq_index_df = pd.read_csv(Path(args.refseq_index), sep='\t')

    # ================= Compile into JSON.
    logger.info("Creating JSON entries.")
    blast_result_dir = output_path.parent / "blast_results"
    min_pct_idty = args.min_pct_idty

    object_entries = create_chronostrain_db(
        blast_result_dir=blast_result_dir,
        strain_df=refseq_index_df,
        gene_paths=ref_gene_paths,
        blast_db_dir=Path(args.blast_db_dir),
        blast_db_name=args.blast_db_name,
        min_pct_idty=min_pct_idty,
        max_target_seqs=args.max_target_seqs
    )

    print_summary(object_entries, ref_gene_paths)
    with open(output_path, 'w') as outfile:
        json.dump(object_entries, outfile, indent=4)
    logger.info(f"Wrote output to {str(output_path)}.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        raise
