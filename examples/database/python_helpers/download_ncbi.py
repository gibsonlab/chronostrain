from typing import List, Tuple, Dict
from pathlib import Path
import json
import click
import os
import time

import pandas as pd
from Bio import SeqIO

from chronostrain import create_logger
from chronostrain.util.io import read_seq_file

logger = create_logger("chronostrain.download_ncbi")


class NCBIRecord:
    def __init__(self, accession: str, genus: str, species: str, strain_name: str):
        self.accession = accession
        self.genus = genus
        self.species = species
        self.strain_name = strain_name


class InvalidQueryError(BaseException):
    pass


def fetch_catalog(taxids: List[str], save_dir: Path, level: str, reference_only: bool, assembly_source: str) -> Dict[str, List[NCBIRecord]]:
    logger.info(f"Got reference only = {reference_only}. Fetching reference genomes only.")
    records = {}
    for taxid in taxids:
        logger.info(f"Handling taxid {taxid}")

        taxid_file_safe = taxid.replace(' ', '_').replace('[', '').replace(']', '')
        save_path = save_dir / f'catalog.{taxid_file_safe}.json'
        try:
            records[taxid] = fetch_catalog_single_taxa(
                taxid,
                save_path,
                level=level,
                assembly_source=assembly_source,
                reference_only=reference_only
            )
        except InvalidQueryError:
            logger.error(f"[*] Something went wrong with TAXID {taxid} query.")
            logger.info(f"Creating error file for {taxid}.")
            (save_path.parent / f"{taxid}.ERROR").touch()
    return records


def fetch_catalog_single_taxa(taxid: str, save_path: Path, level: str, assembly_source: str, reference_only: bool) -> List[NCBIRecord]:
    if reference_only:
        cmd = f"datasets summary genome taxon \"{taxid}\" --assembly-source {assembly_source} "\
              f"--assembly-version latest --assembly-level {level} --exclude-atypical --mag exclude "\
              f"--reference"\
              f"> {save_path}"
    else:
        cmd = f"datasets summary genome taxon \"{taxid}\" --assembly-source {assembly_source} "\
              f"--assembly-version latest --assembly-level {level} --exclude-atypical --mag exclude "\
              f"> {save_path}"
    logger.info("[EXECUTE] {}".format(cmd))
    os.system(cmd)

    if save_path.stat().st_size == 0:
        raise InvalidQueryError()
    with open(save_path, "rt") as f:
        catalog = json.load(f)

    n_entries = catalog['total_count']
    if n_entries == 0:
        return []

    seen_accessions = set()
    records = []

    try:
        logger.info("Found {} raw records.".format(n_entries))
        for entry in catalog['reports']:
            acc = entry['accession']
            if acc in seen_accessions:  # sometimes there are duplicates
                continue
            if 'paired_accession' in entry and entry['paired_accession'] in seen_accessions:
                continue

            org_name = entry['organism']['organism_name']
            name_tokens = org_name.split()[:2]
            genus = name_tokens[0]
            species = ' '.join(name_tokens[1:])

            if genus.startswith("["):
                genus = genus[1:]
            if genus.endswith("]"):
                genus = genus[:-1]
            if species.startswith("sp."):
                print(f"[**] Found special taxon {genus} {species}")

            if 'infraspecific_names' in entry['organism']:
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
            elif entry['source_database'] == 'SOURCE_DATABASE_REFSEQ':
                logger.info("Found record ({}) which is a reference sequence for {} {}.".format(
                    acc, genus, species
                ))
                strain_name = "{}_{}_Ref".format(genus, species)
            else:
                for k, v in entry.items():
                    print(f'{k} --> {v}')
                raise KeyError("Ran into edge case while parsing NCBI datasets JSON file. Parser logic will need to be updated.")

            seen_accessions.add(acc)
            records.append(NCBIRecord(acc, genus, species, strain_name))
    except KeyError as e:
        raise KeyError(f"Couldn't parse NCBI catalog JSON file (`{save_path.name}). Is it properly formatted?") from e
    return records


def download_assemblies(record_list: List[NCBIRecord], out_dir: Path):
    # create input file
    logger.info(f"Downloading assemblies to directory {out_dir}")
    input_file = out_dir / "__ncbi_input.txt"
    with open(input_file, "w") as f:
        for record in record_list:
            print(record.accession, file=f)

    out_zip = out_dir / 'ncbi_dataset.zip'
    cmd = (
        f'datasets download genome accession --inputfile {input_file} '
        f'--include genome,gff3 --assembly-version latest --filename {out_zip}'
    )
    print("[EXECUTE]", cmd)
    os.system(cmd)

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
        print("Indexing {} ({} {}, {})".format(record.accession, record.genus, record.species, record.strain_name))
        seq_dir = data_dir / "ncbi_dataset" / "data" / record.accession
        try:
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
        except ChromosomeNotFoundError:
            print(f"Couldn't find chromosomal sequence for {record.accession}")
    return pd.DataFrame(df_entries)


class ChromosomeNotFoundError(BaseException):
    pass


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
    raise ChromosomeNotFoundError()


@click.command()
@click.option(
    '--taxids', '-t', 'taxids',
    type=str, required=True,
    help="A comma-separated list of taxonomic names (species, genus, etc.)"
)
@click.option(
    '--target-dir', '-d', 'target_dir',
    type=click.Path(path_type=Path, file_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--output-index', '-o', 'output_index_path',
    type=click.Path(path_type=Path, dir_okay=False), required=True
)
@click.option(
    '--level', 'level',
    type=str, required=False, default='complete,chromosome'
)
@click.option(
    '--assembly-source', 'assembly_source',
    type=str, required=False, default='all'
)
@click.option(
    '--reference-only', 'reference_only',
    type=bool, is_flag=True, default=False,
)
def main(taxids: str, target_dir: Path, output_index_path: Path, level: str, reference_only: bool, assembly_source: str):
    records = fetch_catalog(taxids.split(","), save_dir=target_dir, level=level, reference_only=reference_only, assembly_source=assembly_source)

    indices = []
    for tax_id, record_list in records.items():
        if len(record_list) == 0:
            logger.info("Found no records for {}; skipping.".format(tax_id))
            continue
        else:
            logger.info("[{}] Found {} unique strain records.".format(tax_id, len(record_list)))

        time.sleep(1)
        download_assemblies(record_list, target_dir)
        indices.append(create_catalog(record_list, target_dir))

    if len(indices) > 0:
        index_df = pd.concat(indices, ignore_index=True)
        index_df.to_csv(output_index_path, sep='\t', index=False)
    else:
        logger.info("Outputting empty dataframe.")
        output_index_path.touch()


if __name__ == "__main__":
    main()
