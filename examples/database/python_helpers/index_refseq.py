"""
Process and catalog genomes downloaded using the ncbi-genome-download tool.
"""
import argparse
from pathlib import Path
from typing import Tuple, Iterator

import pandas as pd
from Bio import SeqIO

from chronostrain.util.io import read_seq_file
from chronostrain.logging import create_logger
logger = create_logger("chronostrain.index_refseqs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an index file from refseq-derived repository obtained using ncbi-genome-download."
    )

    parser.add_argument('-r', '--refseq_dir', required=True, type=str,
                        help='<Required> The directory containing RefSeq-downloaded files, '
                             'in human-readable directory structure.')

    parser.add_argument('-o', '--output_path', required=False, type=str, default="",
                        help='<Optional> The path to the target output chronostrain db json file.'
                             'Otherwise, defaults to <refseq_dir>/index.tsv.')
    return parser.parse_args()


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
                for fpath in strain_dir.glob("*_genomic.fna.gz"):
                    if fpath.name.endswith('_cds_from_genomic.fna.gz'):
                        continue

                    if fpath.name.endswith('_rna_from_genomic.fna.gz'):
                        continue

                    for accession, assembly_gcf, chrom_path, gff_path, chrom_length in extract_chromosomes(fpath):
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
                            "SeqPath": chrom_path,
                            "ChromosomeLen": chrom_length,
                            "GFF": gff_path
                        })
    return pd.DataFrame(df_entries)


def extract_chromosomes(path: Path) -> Iterator[Tuple[str, str, Path, Path, int]]:
    assembly_gcf = "_".join(path.name.split("_")[:2])
    for record in read_seq_file(path, file_format='fasta'):
        desc = record.description
        accession = record.id.split(' ')[0]

        if "plasmid" in desc or len(record.seq) < 500000:
            continue
        else:
            chrom_path = path.parent / f"{accession}.chrom.fna"
            SeqIO.write([record], chrom_path, "fasta")

            gff_files = list(chrom_path.parent.glob(f"{assembly_gcf}_*_genomic.gff.gz"))
            if len(gff_files) == 0:
                raise RuntimeError(f"No annotations found for {accession} (assembly: {assembly_gcf}).")
            elif len(gff_files) > 1:
                raise RuntimeError(f"Multiple annotation files found for {accession} (assembly: {assembly_gcf}).")
            gff_path = Path(gff_files[0])

            yield accession, assembly_gcf, chrom_path.absolute(), gff_path.absolute(), len(record)


def main():
    args = parse_args()

    refseq_dir = Path(args.refseq_dir)
    refseq_index_df = perform_indexing(refseq_dir)

    if args.output_path != "":
        out_path = Path(args.output_path)
    else:
        out_path = refseq_dir / "index.tsv"

    refseq_index_df.to_csv(out_path, sep='\t', index=False)
    logger.info(f"Wrote index to {str(out_path)}.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        raise
