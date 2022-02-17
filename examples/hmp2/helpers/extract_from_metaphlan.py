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

from typing import Dict, Tuple, Iterator

from chronostrain import create_logger, cfg
logger = create_logger("chronostrain.extract_from_metaphlan")


READ_LEN = 150


def metaphlan_marker_entries(metaphlan_db: Dict, genus: str, species: str) -> Iterator[Tuple[str, Dict]]:
    clade_str = f"s__{genus}_{species}"
    for marker_key, marker_entry in metaphlan_db['markers'].items():
        if clade_str in marker_entry['taxon']:
            yield marker_key, marker_entry


def fetch_and_save_reference(gene_id: str, fasta_path: Path, out_path: Path):
    gene_record = None
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id == gene_id:
            gene_record = record

    if gene_record is None:
        raise RuntimeError(f"Marker gene `{gene_id}` not found in target fasta file `{fasta_path}`.")

    SeqIO.write([gene_record], out_path, "fasta")
    logger.info(f"Wrote gene {gene_id} to {out_path}.")


def extract_reference_marker_genes(
        metaphlan_db: Dict,
        metaphlan_fasta_path: Path,
        genus: str,
        species: str
):
    out_dir = cfg.database_cfg.data_dir / "reference" / genus / species
    logger.info(f"Species {genus} {species}, Target directory: {out_dir}")
    out_dir.mkdir(exist_ok=True, parents=True)

    for metaphlan_gene_id, marker_entry in metaphlan_marker_entries(metaphlan_db, genus, species):
        fetch_and_save_reference(
            metaphlan_gene_id,
            metaphlan_fasta_path,
            out_dir / f"{metaphlan_gene_id}.fasta"
        )


def extract_all_markers(refseq_dir: Path, metaphlan_db: Dict, metaphlan_fasta_path: Path):
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

            extract_reference_marker_genes(metaphlan_db, metaphlan_fasta_path, genus=genus, species=species)


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
    metaphlan_pkl_path = Path(args.metaphlan_pkl_path)
    metaphlan_fasta_path = metaphlan_pkl_path.with_suffix('.fna')

    metaphlan_db = pickle.load(bz2.open(metaphlan_pkl_path, 'r'))
    extract_all_markers(refseq_dir, metaphlan_db, metaphlan_fasta_path)


if __name__ == "__main__":
    main()
