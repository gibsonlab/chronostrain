from typing import Dict
from pathlib import Path
import gzip
import click
import pandas as pd
from urllib import error as urlerror, request as urlrequest, parse as urlparse

from chronostrain.config import cfg
from chronostrain.util.filesystem import convert_size
from chronostrain import create_logger
logger = create_logger('script.initialize_db')


def fetch_post(url: str, data: Dict[str, str], headers: Dict[str, str]) -> str:
    try:
        data_str = urlparse.urlencode(data)
        logger.debug(f"Fetching URL resource via POST: {url}, data={data_str}")
        request = urlrequest.Request(
            url,
            data=data_str.encode(),
            headers=headers
        )
        response = urlrequest.urlopen(request)
    except urlerror.HTTPError as e:
        raise RuntimeError("Failed to retrieve from resource {} due to HTTP error {}.".format(
            url, e.code
        ))
    except urlerror.URLError as e:
        raise RuntimeError("Failed to retrieve from resource {} due to error. Reason: {}.".format(
            url, e.reason
        ))

    r_raw = response.read()
    logger.debug("Got a response of size {}.".format(
        convert_size(len(r_raw))
    ))
    return r_raw.decode('utf-8')


def download_fasta(accession: str, out_path: Path, do_gzip: bool):
    url = 'https://www.ebi.ac.uk/ena/browser/api/fasta'
    if do_gzip:
        f = gzip.open(out_path, 'wt')
    else:
        f = open(out_path, 'wt')

    try:
        f.write(
            fetch_post(
                url=url,
                data={'accessions': accession},
                headers={'accept': 'text/plain'}
            )
        )
    finally:
        f.close()


@click.command()
@click.option(
    '--assembly_tsv', '-a', 'assembly_tsv_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the ENA-derived Isolate Assembly TSV file."
)
@click.option(
    '--reads_tsv', '-r', 'reads_tsv_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the ENA-derived Reads TSV file."
)
@click.option(
    '--participant', '-p', 'target_participant',
    type=str,
    required=True,
    help="The participant ID."
)
@click.option(
    '--outdir', '-o', 'output_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Path to the ENA-derived Reads TSV file."
)
def main(
        assembly_tsv_path: Path,
        reads_tsv_path: Path,
        target_participant: str,
        output_dir: Path
):
    target_download_dir = output_dir / target_participant / 'isolate_assemblies'
    target_download_dir.mkdir(exist_ok=True, parents=True)

    assembly_df = pd.read_csv(assembly_tsv_path, sep='\t').rename(
        columns={'accession': 'assembly_accession'}
    )
    reads_df = pd.read_csv(reads_tsv_path, sep='\t')
    merged_df = assembly_df.merge(
        reads_df, on='sample_accession', how='inner'
    )

    metadata_df = download_for_patient(merged_df, target_participant, target_download_dir)
    metadata_path = target_download_dir / 'metadata.tsv'
    metadata_df.to_csv(metadata_path, sep='\t', index=False)
    initialize_db(metadata_path, db_name=f"P-{target_participant}")


def download_for_patient(sample_df: pd.DataFrame, target_participant: str, target_dir: Path):
    metadata_df_entries = []
    for _, row in sample_df.loc[sample_df['sample_title'].str.startswith(target_participant), :].iterrows():
        assembly_acc = row['assembly_accession']
        genus, species = row['scientific_name'].split()
        sample_title = row['sample_title']

        tokens = sample_title.split('_')
        if len(tokens) == 2:
            timepoint = tokens[1]
            sample_id = '*'
        elif len(tokens) == 3:
            timepoint = tokens[1]
            sample_id = tokens[2]
        else:
            raise ValueError(f"Was unable to parse the sample title `{sample_title}`.")

        fasta_path = target_dir / f'{assembly_acc}.fasta'
        logger.info(f"Downloading assembly {assembly_acc} [{genus} {species}]")
        download_fasta(assembly_acc, fasta_path, do_gzip=False)
        metadata_df_entries.append({
            'Participant': target_participant,
            'Accession': assembly_acc,
            'FastaPath': fasta_path,
            'Genus': genus,
            'Species': species,
            'Timepoint': timepoint,
            'SampleId': sample_id
        })
    return pd.DataFrame(metadata_df_entries)


def initialize_db(metadata_path: Path, db_name: str):
    from chronostrain.database import IsolateAssemblyDatabase
    _ = IsolateAssemblyDatabase(
        db_name=db_name,
        specs=metadata_path,
        data_dir=cfg.database_cfg.data_dir
    )


if __name__ == "__main__":
    main()
