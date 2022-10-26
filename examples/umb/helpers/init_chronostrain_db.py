"""
Script which creates a chronostrain db of all UNIREF-specified marker genes.
Works in 3 steps:
    1) For each marker gene name, extract its annotation from reference K-12 genome.
    2) Run BLAST to find all matching hits (even partial hits) of marker gene sequence to each strain assembly
        (with blastn configured to allow, on average, up to 10 hits per genome).
    3) Convert BLAST results into chronostrain JSON database specification.
"""
from abc import ABC
import argparse
from collections import defaultdict
from pathlib import Path
import csv
import json

import pandas as pd
import pickle
import bz2

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation
from Bio.SeqRecord import SeqRecord
from bioservices import UniProt

from chronostrain.util.entrez import fetch_genbank
from chronostrain.util.external import blastn

from typing import List, Set, Dict, Any, Tuple, Iterator

from chronostrain.util.io import read_seq_file
from chronostrain.config import cfg
from chronostrain.logging import create_logger

logger = create_logger("chronostrain.init_db")


READ_LEN = 150
_BLAST_OUT_FMT = "6 saccver sstart send slen qstart qend evalue pident gaps qcovhsp"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create chronostrain database file from specified uniprot marker CSV."
    )

    # Input specification.
    parser.add_argument('-u', '--uniprot_csv', required=False, type=str, default='',
                        help='<Optional> A path to a two-column CSV file (<UniprotID>, <ClusterName>) format specifying'
                             'any desired additional genes not given by metaphlan.')
    parser.add_argument('-m', '--metaphlan_pkl', required=False, action='append',
                        help='<Optional> A path to the metaphlan database pickle file.'
                             'Can be repeated to specify more than one.')
    parser.add_argument('-g', '--genes_fasta', required=False, type=str, default='',
                        help='<Optional> A path to a fasta file listing out genes. Each records ID must be the '
                             'desired gene name.')
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
    parser.add_argument('--reference_accession', required=False, type=str,
                        default='U00096.3',
                        help='<Optional> The reference genome to use for pulling out annotated gene sequences.')
    return parser.parse_args()


# ===================================================== BEGIN Local resources
def run_blast_local(db_dir: Path,
                    db_name: str,
                    result_dir: Path,
                    gene_paths: Dict[str, Path],
                    max_target_seqs: int,
                    min_pct_idty: int,
                    out_fmt: str) -> Dict[str, Path]:
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
        num_ref_genomes: int
) -> List[Dict[str, Any]]:
    """
    :return:
    """

    blast_results = run_blast_local(
        db_dir=blast_db_dir,
        db_name=blast_db_name,
        result_dir=blast_result_dir,
        gene_paths=gene_paths,
        min_pct_idty=min_pct_idty,
        max_target_seqs=10 * num_ref_genomes,
        out_fmt=_BLAST_OUT_FMT
    )

    return create_strain_entries(blast_results, gene_paths, strain_df)


def create_strain_entries(blast_results: Dict[str, Path], ref_gene_paths: Dict[str, Path], strain_df: pd.DataFrame):
    def _entry_initializer(_accession):
        strain_row = strain_df.loc[strain_df['Accession'] == _accession, :].head(1)
        subj_genus = strain_row['Genus'].item()
        subj_species = strain_row['Species'].item()
        subj_strain_name = strain_row['Strain'].item()
        genome_length = strain_row['ChromosomeLen'].item()

        return {
            'id': _accession,
            'genus': subj_genus,
            'species': subj_species,
            'name': subj_strain_name,
            'genome_length': genome_length,
            'seqs': [{'accession': _accession, 'seq_type': 'chromosome'}],
            'markers': []
        }

    strain_entries = {}

    # ===================== Parse BLAST hits.
    for gene_name, blast_result_path in blast_results.items():
        blast_hits = parse_blast_hits(blast_result_path)

        # Parse the entries.
        canonical_gene_found = False
        ref_gene_path = ref_gene_paths[gene_name]
        ref_gene_len = len(next(iter(read_seq_file(ref_gene_path, 'fasta'))).seq)
        min_canonical_length = ref_gene_len * 0.95

        logger.debug(f"Parsing BLAST hits for gene `{gene_name}`.")
        for subj_acc in blast_hits.keys():
            if subj_acc not in strain_entries:
                strain_entries[subj_acc] = _entry_initializer(subj_acc)

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
                """
                BLAST convention: if hit is on the minus strand, start > end (to indicate the negative direction).
                *** IMPORTANT *** --- note that the actual VALUES of (start, end) is with respect to the plus strand.
                """
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


# ======================== Reference sequence loaders ========================================
class GeneLoader(ABC):
    def load_genes(self) -> Iterator[Tuple[str, SeqRecord]]:
        """
        :return: A generator over (gene_name, reference sequence) pairs.
        """
        raise NotImplementedError()


class UniProtLoader(GeneLoader):
    def __init__(self, uniprot_csv: Path, ref_accession: str):
        self.uniprot_csv = uniprot_csv
        self.reference_accession = ref_accession
        self.reference_dir = cfg.database_cfg.data_dir / "reference" / ref_accession
        self.reference_dir.mkdir(exist_ok=True, parents=True)
        self.gb_file = fetch_genbank(self.reference_accession, self.reference_dir)

    def load_genes(self) -> Iterator[Tuple[str, SeqRecord]]:
        logger.info(f"Loading markers from uniprot collection {self.uniprot_csv}.")
        logger.info(f"Downloading reference accession {self.reference_accession}")

        chromosome_seq = next(SeqIO.parse(self.gb_file, "gb")).seq

        clusters_already_found: Set[str] = set()
        clusters_to_find: Set[str] = set()
        gene_cluster_mapping: Dict[str, str] = {}

        for gene_name, gene_cluster in self.get_uniprot_genes():
            clusters_to_find.add(gene_name)
            gene_cluster_mapping[gene_name.lower()] = gene_name
            for other_gene in gene_cluster:
                gene_cluster_mapping[other_gene.lower()] = gene_name

        for gb_gene_name, locus_tag, location in self.parse_genbank_genes():
            if gb_gene_name.lower() not in gene_cluster_mapping:
                continue
            if gene_cluster_mapping[gb_gene_name.lower()] not in clusters_to_find:
                continue

            found_gene = gene_cluster_mapping[gb_gene_name.lower()]
            logger.info(f"Found uniref cluster {found_gene} for accession {self.reference_accession} (name={gb_gene_name})")

            if found_gene in clusters_already_found:
                logger.warning(f"Multiple copies of {found_gene} found in {self.reference_accession}. Skipping second instance.")
            else:
                clusters_already_found.add(found_gene)
                clusters_to_find.remove(found_gene)

                gene_seq = location.extract(chromosome_seq)
                record = SeqRecord(gene_seq, id=f"{found_gene}", description=f"REFERENCE_{self.reference_accession}_{str(location)}")
                yield found_gene, record

        if len(clusters_to_find) > 0:
            logger.warning("Couldn't find uniprot genes {}.".format(
                ",".join(clusters_to_find))
            )
        logger.info(f"Finished parsing reference accession {self.reference_accession}.")

    def parse_uniprot_csv(self) -> Iterator[Tuple[str, str]]:
        with open(self.uniprot_csv, "r") as f:
            for line in f:
                if len(line.strip()) == 0:
                    continue

                tokens = line.strip().split("\t")
                uniprot_id, gene_name, metadata = tokens[0], tokens[1], tokens[2]

                if uniprot_id == "UNIPROT_ID":
                    # Header line
                    continue

                yield uniprot_id, gene_name

    def parse_genbank_genes(self) -> Iterator[Tuple[str, str, FeatureLocation]]:
        for record in SeqIO.parse(self.gb_file, format="gb"):
            for feature in record.features:
                if feature.type == "gene":
                    gene_name = feature.qualifiers['gene'][0]
                    locus_tag = feature.qualifiers['locus_tag'][0]
                    yield gene_name, locus_tag, feature.location

    def get_uniprot_genes(self) -> Iterator[Tuple[str, List[str]]]:
        u = UniProt(verbose=False)

        for uniprot_id, gene_name in self.parse_uniprot_csv():
            logger.info(f"Performing search for Uniprot id {uniprot_id}...")
            res = u.quick_search(uniprot_id)

            if uniprot_id not in res:
                logger.info(
                    f"No result found for UniProt query `{uniprot_id}`."
                )
                continue
            else:
                logger.info(f"Found {len(res)} hits for UniProt query `{uniprot_id}`.")

            cluster = res[uniprot_id]['Gene names'].split()
            yield gene_name, cluster


class MetaphlanLoader(GeneLoader):
    def __init__(self, metaphlan_pkl_path: Path):
        self.pkl_path = metaphlan_pkl_path
        self.marker_fasta = metaphlan_pkl_path.with_suffix('.fna')
        if not self.marker_fasta.exists():
            raise FileNotFoundError(f"Expected {self.marker_fasta} to exist, but not found.")

    def load_genes(self) -> Iterator[Tuple[str, SeqRecord]]:
        logger.info(f"Searching for E.coli markers from MetaPhlAn database: {self.pkl_path.stem}.")
        with bz2.open(self.pkl_path, "r") as f:
            db = pickle.load(f)

        markers = db['markers']
        target_keys = set()
        for marker_key, marker_dict in markers.items():
            if 's__Escherichia_coli' in marker_dict['taxon']:
                target_keys.add(marker_key)

        logger.info(f"Target # of markers: {len(target_keys)}")
        for marker_key, seq in self.retrieve_reference(target_keys):
            logger.info(f"Found metaphlan marker ID {marker_key}.")
            record = SeqRecord(seq, id=marker_key, description=self.pkl_path.stem)
            yield marker_key, record

    def retrieve_reference(self, marker_keys: Set[str]) -> Tuple[str, Seq]:
        remaining = set(marker_keys)
        with open(self.marker_fasta, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                if len(remaining) == 0:
                    break
                if record.id not in remaining:
                    continue

                remaining.remove(record.id)
                yield record.id, record.seq
        if len(remaining) > 0:
            logger.warning(f"For some reason, couldn't locate {len(remaining)} markers: {remaining}")


class FastaLoader(GeneLoader):
    def __init__(self, fasta_path: Path):
        self.fasta_path = fasta_path

    def load_genes(self) -> Iterator[Tuple[str, SeqRecord]]:
        logger.info(f"Loading markers from fasta file {self.fasta_path}.")
        for record in SeqIO.parse(self.fasta_path, 'fasta'):
            gene_name = record.id
            logger.info(f"Found FASTA record for gene `{gene_name}` ({record.description})")
            yield gene_name, record


def retrieve_reference(loaders: List[GeneLoader]) -> Dict[str, Path]:
    target_dir = cfg.database_cfg.data_dir / "reference"
    target_dir.mkdir(exist_ok=True, parents=True)

    gene_paths: Dict[str, Path] = {}

    for loader in loaders:
        for gene_name, record in loader.load_genes():
            if gene_name in gene_paths:
                logger.warning(f"gene `{gene_name}` was found in two separate instances. "
                               f"Skipping loader {loader.__class__.__name__}")

            gene_out_path = target_dir / f"{gene_name}.fasta"
            SeqIO.write(record, gene_out_path, "fasta")
            gene_paths[gene_name] = gene_out_path

    return gene_paths


def main():
    args = parse_args()

    if args.uniprot_csv is None and args.genes_fasta is None and args.metaphlan_pkl is None:
        print("Must specify at least one of --uniprot_path, --metaphlan_pkl, or --primers_path.")
        exit(1)

    loaders = []

    if args.genes_fasta is not None:
        loaders.append(
            FastaLoader(Path(args.genes_fasta))
        )

    if args.uniprot_csv is not None:
        loaders.append(
            UniProtLoader(Path(args.uniprot_csv), args.reference_accession)
        )

    if args.metaphlan_pkl is not None and len(args.metaphlan_pkl) > 0:
        for f in args.metaphlan_pkl:
            loaders.append(
                MetaphlanLoader(Path(f))
            )

    output_path = Path(args.output_path)

    # ================= Pull out reference genes
    logger.info(f"Retrieving reference genes from {args.reference_accession}")
    ref_gene_paths = retrieve_reference(loaders)

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
        num_ref_genomes=refseq_index_df.shape[0]
    )

    with open(output_path, 'w') as outfile:
        json.dump(object_entries, outfile, indent=4)
    logger.info(f"Wrote output to {str(output_path)}.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        raise
