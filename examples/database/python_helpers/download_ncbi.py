from typing import List, Tuple
from pathlib import Path
import json
import click

import pandas as pd
from Bio import SeqIO

from chronostrain import create_logger
from chronostrain.util.external import call_command
from chronostrain.util.io import read_seq_file

logger = create_logger("chronostrain.download_ncbi")

class NCBIRecord:
    def __init__(self, accession: str, genus: str, species: str, strain_name: str):
        self.accession = accession
        self.genus = genus
        self.species = species
        self.strain_name = strain_name


def fetch_catalog(taxid, save_path: Path) -> List[NCBIRecord]:
    call_command(
        "datasets",
        [
            "summary", "genome", "taxon",
            f"\"{taxid}\"",
            '--assembly-version', 'latest',
            "--assembly-level", "complete",
            '--exclude-atypical'
        ],
        stdout=save_path,
        silent=False
    )

    with open(save_path, "rt") as f:
        catalog = json.load(f)

    seen_accessions = set()
    records = []
    for entry in catalog['reports']:
        acc = entry['accession']
        if acc in seen_accessions:  # sometimes there are duplicates
            continue

        org_name = entry['organism']['organism_name']
        genus, species = org_name.split()[:2]
        infra_name = entry['organism']['infraspecific_names']
        if 'strain' in infra_name:
            strain_name = infra_name['strain']
        elif 'isolate' in infra_name:
            strain_name = infra_name['isolate']
        else:
            raise KeyError(
                "No recognizable key found for infraspecific_names dict of {}. (Possible keys: {})".format(
                    acc, set(infra_name.keys())
                )
            )

        records.append(NCBIRecord(acc, genus, species, strain_name))
    return records


def download_assemblies(records: List[NCBIRecord], out_dir: Path):
    # create input file
    input_file = out_dir / "__ncbi_input.txt"
    with open(input_file, "w") as f:
        for record in records:
            print(record.accession, file=f)

    out_zip = out_dir / 'ncbi_dataset.zip'
    call_command(
        "datasets",
        [
            "download", "genome", "accession",
            '--inputfile', input_file,
            '--include', 'genome,gff3',
            '--assembly-version', 'latest',
            '--filename', out_zip
        ]
    )

    # unzip
    import zipfile
    with zipfile.ZipFile(out_zip, "r") as zip_ref:
        zip_ref.extractall(out_dir)

    # clean up
    input_file.unlink()
    out_zip.unlink(missing_ok=True)


def create_catalog(records: List[NCBIRecord], data_dir: Path) -> pd.DataFrame:
    df_entries = []
    for record in records:
        print("Indexing {} ({})".format(record.accession, record.strain_name))
        seq_dir = data_dir / "ncbi_dataset" / "read_frags" / record.accession
        nuc_accession, chrom_path, gff_path, chrom_len = extract_chromosome(seq_dir)

        df_entries.append({
            "Genus": record.genus,
            "Species": record.species,
            "Strain": record.strain_name,
            "Accession": nuc_accession,
            "Assembly": record.accession,
            "SeqPath": chrom_path,
            "ChromosomeLen": chrom_len,
            "GFF": gff_path,
        })
    return pd.DataFrame(df_entries)

def extract_chromosome(seq_dir: Path) -> Tuple[str, Path, Path, int]:
    fasta_path = next(iter(seq_dir.glob("*.fna")))
    gff_path = seq_dir / "genomic.gff"

    for record in read_seq_file(fasta_path, file_format='fasta'):
        desc = record.description
        nuc_accession = record.id.split(' ')[0]

        if "plasmid" in desc or len(record.seq) < 500000:
            continue
        else:
            chrom_path = seq_dir / f"{nuc_accession}.chrom.fna"
            SeqIO.write([record], chrom_path, "fasta")
            return nuc_accession, chrom_path.absolute(), gff_path.absolute(), len(record)

@click.command()
@click.option(
    '--taxid', '-t', 'taxid',
    type=str, required=True
)
@click.option(
    '--target-dir', '-d', 'target_dir',
    type=click.Path(path_type=Path, file_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--output-index', '-o', 'output_index_path',
    type=click.Path(path_type=Path, dir_okay=False), required=True
)
def main(taxid: str, target_dir: Path, output_index_path: Path):
    records = fetch_catalog(taxid, save_path=target_dir / "catalog.json")
    logger.info("Found {} records.".format(len(records)))
    download_assemblies(records, target_dir)
    index_df = create_catalog(records, target_dir)
    index_df.to_csv(output_index_path, sep='\t', index=False)


if __name__ == "__main__":
    main()
