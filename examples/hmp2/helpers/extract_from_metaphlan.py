"""
Script which creates a chronostrain db of all UNIREF-specified marker genes.
Works in 3 steps:
    1) For each marker gene name, extract its annotation from reference K-12 genome.
    2) Run BLAST to find all matching hits (even partial hits) of marker gene sequence to each strain assembly
        (with blastn configured to allow, on average, up to 10 hits per genome).
    3) Convert BLAST results into chronostrain JSON database specification.
"""
import argparse
import pickle
import bz2
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from bioservices import UniProt

from typing import List, Dict, Tuple, Iterator, Set

from chronostrain import create_logger, cfg
logger = create_logger("chronostrain.extract_from_metaphlan")


READ_LEN = 150

class UniprotError(BaseException):
    pass


class EntrezError(BaseException):
    def __init__(self, query: str):
        super().__init__()
        self.query = query


def metaphlan_marker_entries(metaphlan_db: Dict, genus: str, species: str) -> Iterator[Tuple[str, Dict]]:
    clade_str = f"s__{genus}_{species}"
    for marker_key, marker_entry in metaphlan_db['markers'].items():
        if clade_str in marker_entry['taxon']:
            yield marker_key, marker_entry


def get_reference_sequence(gene_id: str, reference_acc: List[str], gene_names: Set[str]) -> SeqRecord:
    query = "({}) AND ({})".format(
        " OR ".join(f"{acc}[Primary Accession]" for acc in reference_acc),
        " OR ".join(f"{g}[Gene Name]" for g in gene_names)
    )

    logger.debug(f"Entrez query: `{query}`")
    handle = Entrez.esearch("nucleotide", query)
    records = Entrez.read(handle)
    handle.close()

    if len(records) == 0 or len(records['IdList']) == 0:
        raise EntrezError(query)

    handle = Entrez.efetch(db="nucleotide", id=records['IdList'][0], rettype="gb", retmode="text")

    try:
        record = SeqIO.read(handle, "genbank")

        logger.debug(f"Found genbank record {record.id}")
        genome = record.seq
        for feature in record.features:
            if feature.type != 'gene':
                continue

            locus_tag = feature.qualifiers.get('locus_tag', [''])[0]
            feature_gene_name = feature.qualifiers.get('gene', [''])[0]

            if (locus_tag not in gene_names) and (feature_gene_name not in gene_names):
                continue

            record = SeqRecord(
                seq=Seq(feature.extract(genome)),
                id=f"REF_{gene_id}",
                description=f"{record.id}_{str(feature.location)}"
            )
            return record
        raise RuntimeError(f"Couldn't parse gene {gene_id} from entrez id {record.id}")
    finally:
        handle.close()


def get_uniprot_references(u: UniProt, uniprot_id: str, genus: str, out_dir: Path) -> str:
    query = f"{uniprot_id}+AND+{genus}"
    logger.debug(f"Uniprot query: `{query}`")
    res: str = u.search(query, frmt='tab', columns="id,genes,database(EMBL)", maxTrials=10)
    lines = res.strip().split('\n')

    if len(lines) <= 1:
        raise UniprotError()
    else:
        lines = lines[1:]
        logger.debug(f"Found {len(lines)} hits for UniProt query `{uniprot_id}`.")

    line = lines[0]
    gene_name, cluster, embl_refs = line.split('\t')
    gene_out_path = out_dir / f"{gene_name}.fasta"

    cluster = set(cluster.split(' '))
    embl_refs = embl_refs.split(';')
    if len(embl_refs[-1]) == 0:
        embl_refs = embl_refs[:-1]
    if len(embl_refs) == 0:
        raise ValueError(f"No EMBL references for uniprot entry `{uniprot_id}`")

    if gene_out_path.exists():
        logger.info(f"File {gene_out_path} already found.")
    else:
        record = get_reference_sequence(gene_name, embl_refs, cluster)
        SeqIO.write(
            record,
            gene_out_path,
            "fasta"
        )
    return gene_name


def extract_reference_marker_genes(
        metaphlan_db: Dict,
        genus: str,
        species: str
):
    out_dir = cfg.database_cfg.data_dir / "reference" / genus / species
    logger.info(f"Target directory: {out_dir}")
    out_dir.mkdir(exist_ok=True, parents=True)

    u = UniProt()

    for marker_key, marker_entry in metaphlan_marker_entries(metaphlan_db, genus, species):
        tokens = marker_key.split('__')
        uniprot_id = tokens[1]

        logger.info(f"Found uniprot ID {uniprot_id} for {genus} {species} (metaphlan token was `{marker_key}`).")
        try:
            gene_name = get_uniprot_references(u, uniprot_id, genus, out_dir)
            logger.info(f"Gene found: Uniprot = {uniprot_id}, name = {gene_name}")
        except UniprotError:
            logger.debug(
                f"No result found for UniProt query `{uniprot_id}`, derived from metaphlan ID `{marker_key}`."
            )
            continue
        except EntrezError as e:
            logger.debug(
                f"Failed to find Entrez entries {uniprot_id}. Query was `{e.query}`."
            )
            continue


def extract_all_markers(refseq_dir: Path, metaphlan_db: Dict):
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
            if species == 'sp.':
                continue

            extract_reference_marker_genes(metaphlan_db, genus=genus, species=species)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract marker genes and reference sequences based on Metaphlan database."
    )

    # Input specification.
    parser.add_argument('-m', '--metaphlan_pkl_path', required=True, type=str,
                        help='<Required> The path to the metaphlan pickle database file.')
    parser.add_argument('-r', '--refseq_dir', required=True, type=str,
                        help='<Required> The strainGE database directory.')
    return parser.parse_args()


def main():
    args = parse_args()

    Entrez.email = cfg.entrez_cfg.email
    logger.info(f"Configured Entrez to use email `{cfg.entrez_cfg.email}`.")

    refseq_dir = Path(args.refseq_dir)
    metaphlan_db = pickle.load(bz2.open(args.metaphlan_pkl_path, 'r'))
    extract_all_markers(refseq_dir, metaphlan_db)


if __name__ == "__main__":
    main()
