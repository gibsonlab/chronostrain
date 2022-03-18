"""
Script which creates a chronostrain db of all UNIREF-specified marker genes.
Works in 3 steps:
    1) For each marker gene name, extract its annotation from reference K-12 genome.
    2) Run BLAST to find all matching hits (even partial hits) of marker gene sequence to each strain assembly
        (with blastn configured to allow, on average, up to 10 hits per genome).
    3) Convert BLAST results into chronostrain JSON database specification.
"""
import os
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
import csv
import json
import pickle
import bz2
import tarfile
import pandas as pd

import shutil
import urllib.request as request
from contextlib import closing

from Bio import SeqIO
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create chronostrain database file from specified gene_info_uniref marker CSV."
    )

    # Input specification.
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='<Required> The path to the target output chronostrain db json file.')

    # Optional params
    parser.add_argument('--use_local', action='store_true', type=bool, required=False,
                        help='If flag is set, use locally stored RefSeq files. User must specify the directory via the '
                             '-r option.')
    parser.add_argument('-r', '--refseq_dir', required=False, type=str, default="",
                        help='<Optional> The directory containing RefSeq-downloaded files, '
                             'in human-readable directory structure.')
    parser.add_argument('--min_pct_idty', required=False, type=int, default=50,
                        help='<Optional> The percent identity threshold for BLAST. (default: 50)')
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

                    for accession, assembly_gcf, chrom_path in extract_chromosomes(fpath):
                        logger.debug("Found accession {} from assembly {} ({} {}, Strain `{}`)".format(
                            accession,
                            assembly_gcf,
                            genus,
                            species,
                            strain_name
                        ))
                        df_entries.append({
                            "Genus": genus,
                            "Species": species,
                            "Strain": strain_name,
                            "Accession": accession,
                            "Assembly": assembly_gcf,
                            "SeqPath": chrom_path
                        })
    return pd.DataFrame(df_entries)


def extract_chromosomes(path: Path) -> Iterator[Tuple[str, Path]]:
    assembly_gcf = "_".join(path.name.split("_")[:2])
    for record in read_seq_file(path, file_format='fasta'):
        desc = record.description
        accession = record.id.split(' ')[0]

        if "plasmid" in desc or len(record.seq) < 500000:
            continue
        else:
            chrom_path = path.parent / f"{accession}.chrom.fna"
            SeqIO.write([record], chrom_path, "fasta")

            yield accession, assembly_gcf, chrom_path


def run_blast_local(blast_db_dir: Path,
                    blast_result_dir: Path,
                    gene_paths: Dict[str, Path],
                    strain_fasta_files: List[Path],
                    max_target_seqs: int,
                    min_pct_idty: int,
                    out_fmt: str) -> Dict[str, Path]:
    blast_db_name = "db_esch"
    blast_db_title = "\"Escherichia (metaphlan markers, NCBI complete)\""
    blast_fasta_path = blast_db_dir / "genomes.fasta"

    # ========= Initialize BLAST database.
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

    # Run BLAST.
    blast_results: Dict[str, Path] = {}
    blast_result_dir.mkdir(parents=True, exist_ok=True)
    for gene_name, ref_gene_path in gene_paths.items():
        logger.info(f"Running blastn on {gene_name}.")
        blast_result_path = blast_result_dir / f"{gene_name}.tsv"
        blastn(
            db_name=blast_db_name,
            db_dir=blast_db_dir,
            query_fasta=ref_gene_path,
            perc_identity_cutoff=min_pct_idty,
            out_path=blast_result_path,
            num_threads=cfg.model_cfg.num_cores,
            out_fmt=out_fmt,
            max_target_seqs=max_target_seqs,
            strand="both"
        )
        blast_results[gene_name] = blast_result_path
    return blast_results
# ===================================================== END Local resources


# ===================================================== BEGIN Remote resources
def run_blast_remote(gene_paths: Dict[str, Path],
                     blast_result_dir: Path,
                     max_target_seqs: int,
                     min_pct_idty: int,
                     out_fmt: str) -> Dict[str, Path]:
    blast_result_dir.mkdir(parents=True, exist_ok=True)

    from Bio.Blast.Applications import NcbiblastnCommandline
    import subprocess

    blast_results: Dict[str, Path] = {}
    for gene_name, gene_path in gene_paths.items():
        cline = NcbiblastnCommandline(
            db='nt',
            num_alignments=max_target_seqs,
            perc_identity=min_pct_idty,
            outfmt=out_fmt,
            strand="both",
            query=str(gene_path)
        )

        output_path = blast_result_dir / f"{gene_name}.tsv"
        output_file = open(output_path, 'w')
        p = subprocess.run(
            str(cline),
            stdout=output_file,
            stderr=subprocess.PIPE
        )

        if p.returncode != 0:
            stderr = p.stderr.decode("utf-8").strip()
            raise RuntimeError(f"BLAST returned with nonzero return code `{p.returncode}`. STDERR was: `{stderr}`")
        else:
            blast_results[gene_name] = output_path
    return blast_results


# ===================================================== END Remote resources


# ============= Rest of initialization
def create_chronostrain_db_local(
        blast_result_dir: Path,
        gene_paths: Dict[str, Path],
        seq_dir: Path,
        min_pct_idty: int,
        taxonomy: 'Taxonomy'
) -> List[Dict[str, Any]]:
    """
    :return:
    """
    # ================= Indexing of refseqs
    logger.info(f"Indexing refseqs located in {seq_dir}")
    seq_index = perform_indexing(seq_dir)
    index_path = seq_dir / "index.tsv"
    seq_index.to_csv(index_path, sep='\t', index=False)
    logger.info(f"Wrote index to {str(index_path)}.")

    # ======== BLAST configuration
    blast_db_dir = blast_result_dir / "blast_db"
    logger.info("BLAST\n\tdatabase location: {}\n\tresults directory: {}".format(
        str(blast_db_dir),
        str(blast_result_dir)
    ))

    strain_fasta_files = []
    for idx, row in seq_index.iterrows():
        accession = row['Accession']
        strain_id = accession  # Use the accession as the strain's ID (we are only using full chromosome assemblies).

        fasta_path = row['SeqPath']
        strain_fasta_files.append(fasta_path)
        symlink_path = strain_seq_dir(strain_id) / f"{accession}.fasta"
        if symlink_path.exists():
            logger.info(f"Path {symlink_path} already exists.")
        else:
            symlink_path.parent.mkdir(exist_ok=True, parents=True)
            symlink_path.symlink_to(fasta_path)
            logger.info(f"Symlink {symlink_path} -> {fasta_path}")

    blast_results = run_blast_local(
        blast_db_dir,
        blast_result_dir,
        gene_paths,
        strain_fasta_files,
        max_target_seqs=10 * seq_index.shape[0],
        min_pct_idty=min_pct_idty,
        out_fmt=_BLAST_OUT_FMT
    )

    return create_strain_entries(blast_results, gene_paths, taxonomy)


def create_chronostrain_db_remote(
        blast_result_dir: Path,
        gene_paths: Dict[str, Path],
        min_pct_idty: int,
        taxonomy: 'Taxonomy'
) -> List[Dict[str, Any]]:
    blast_results = run_blast_remote(
        gene_paths,
        blast_result_dir,
        max_target_seqs=5000,  # May need to raise this?
        min_pct_idty=min_pct_idty,
        out_fmt=_BLAST_OUT_FMT
    )

    return create_strain_entries(blast_results, gene_paths, taxonomy)


def blank_strain_entry(strain_id: str, genus: str, species: str, name: str, accession: str):
    return {
        'id': strain_id,
        'genus': genus,
        'species': species,
        'name': name,
        'seqs': [{'accession': accession, 'seq_type': 'chromosome'}],
        'markers': []
    }


def create_strain_entries(blast_results: Dict[str, Path], ref_gene_paths: Dict[str, Path], taxonomy):
    strain_entries: Dict[str, Dict] = {}

    for gene_name, blast_result_path in blast_results:
        blast_hits = parse_blast_hits(blast_result_path, taxonomy)

        # Parse the entries.
        canonical_gene_found = False
        ref_gene_path = ref_gene_paths[gene_name]
        ref_gene_len = len(next(iter(read_seq_file(ref_gene_path, 'fasta'))).seq)
        min_canonical_length = ref_gene_len * 0.95

        logger.debug(f"Parsing BLAST hits for gene `{gene_name}`.")
        for subj_acc in blast_hits.keys():
            # Create strain entries if they don't already exist.
            if subj_acc not in strain_entries:
                sample_hit = blast_hits[subj_acc][0]
                strain_entries[subj_acc] = blank_strain_entry(
                    strain_id=subj_acc,
                    genus=sample_hit.subj_genus,
                    species=sample_hit.subj_species,
                    name=sample_hit.subj_sci_name,
                    accession=subj_acc
                )

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


_BLAST_OUT_FMT = "6 saccver sstart send slen qstart qend evalue bitscore pident gaps qcovhsp staxids"


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
                 bitscore: float,
                 pct_identity: float,
                 num_gaps: int,
                 query_coverage_per_hsp: float,
                 subj_taxid: str,
                 subj_genus: str,
                 subj_species: str,
                 subj_sci_name: str
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
        self.bitscore = bitscore
        self.pct_identity = pct_identity
        self.num_gaps = num_gaps
        self.query_coverage_per_hsp = query_coverage_per_hsp
        self.subj_taxid = subj_taxid
        self.subj_genus = subj_genus
        self.subj_species = subj_species
        self.subj_sci_name = subj_sci_name


class Taxonomy(object):
    def __init__(self, taxdump_tar: Path):
        self.mapping: Dict[str, Tuple[str, str, str]] = {}
        tar = tarfile.open(taxdump_tar)
        f = tar.extractfile("names.dmp")
        for line_idx, line in enumerate(f):
            line = line.decode("utf-8").strip()
            tax_id, name_txt, unique_name, name_class, _ = [token.strip() for token in line.split("|")]

            name_tokens = name_txt.split()
            if len(name_tokens) < 3:
                continue
            genus = name_tokens[0].strip()
            species = name_tokens[1].strip()
            name = ' '.join(name_tokens[2:])
            self.mapping[tax_id] = (genus, species, name)
        tar.close()

    def get(self, staxid: str) -> Tuple[str, str, str]:
        """
        Returns a (genus, species, subj_name) triple.
        """
        return self.mapping[staxid]


def parse_blast_hits(blast_result_path: Path, taxonomy: Taxonomy) -> Dict[str, List[BlastHit]]:
    accession_to_positions: Dict[str, List[BlastHit]] = defaultdict(list)
    with open(blast_result_path, "r") as f:
        blast_result_reader = csv.reader(f, delimiter='\t')
        for row_idx, row in enumerate(blast_result_reader):
            subj_acc, subj_start, subj_end, subj_len, qstart, qend, evalue, bitscore, pident, gaps, qcovhsp, staxid = row

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

            genus, species, subj_sci_name = taxonomy.get(staxid)

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
                    bitscore=float(bitscore),
                    pct_identity=float(pident),
                    num_gaps=int(gaps),
                    query_coverage_per_hsp=float(qcovhsp),
                    subj_taxid=staxid,
                    subj_genus=genus,
                    subj_species=species,
                    subj_sci_name=subj_sci_name
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
            logger.debug(f"Searching additional cluster `{cluster_name}`, uniprot ID `{uniprot_id}`")
            yield uniprot_id


def download_taxonomies(target_dir: Path) -> Taxonomy:
    target_url = "ftp://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.Z"
    target_dir.mkdir(exist_ok=True, parents=True)
    target_file = target_dir / "taxdump.tar.Z"

    with closing(request.urlopen(target_url)) as r:
        with open(target_file, 'wb') as f:
            shutil.copyfileobj(r, f)

    os.system(f'uncompress {target_file}')
    return Taxonomy(target_dir / "taxdump.tar")


def main():
    args = parse_args()
    output_path = Path(args.output_path)

    # ================= Pull out reference genes
    logger.info(f"Retrieving reference genes from {args.reference_accession}")
    ref_gene_paths = download_reference(args.reference_accession, args.metaphlan_pkl_path, args.uniprot_csv)

    # ================= Compile into JSON.
    logger.info("Creating JSON entries.")
    taxonomy = download_taxonomies(output_path.parent / "taxonomy")
    blast_result_dir = output_path.parent / "blast_results"
    min_pct_idty = args.min_pct_idty
    if args.mode == "local":
        if args.refseq_dir == "":
            raise ValueError("In local mode, refseq_dir must be specified.")
        else:
            seq_dir = Path(args.refseq_dir)
        object_entries = create_chronostrain_db_local(
            blast_result_dir, ref_gene_paths, seq_dir, min_pct_idty, taxonomy
        )
    else:
        object_entries = create_chronostrain_db_remote(
            blast_result_dir, ref_gene_paths, min_pct_idty, taxonomy
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
