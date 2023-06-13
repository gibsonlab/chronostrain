from typing import Dict
from pathlib import Path
import gzip
import click
import pandas as pd
from urllib import error as urlerror, request as urlrequest, parse as urlparse
import math


def convert_size(size_bytes: int) -> str:
    """
    Converts bytes to the nearest useful meaningful unit (B, KB, MB, GB, etc.)
    Code credit to https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python/14822210
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def fetch_post(url: str, data: Dict[str, str], headers: Dict[str, str]) -> str:
    try:
        data_str = urlparse.urlencode(data)
        print(f"Fetching URL resource via POST: {url}, data={data_str}")
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
    print("Got a response of size {}.".format(convert_size(len(r_raw))))
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
    '--outdir', '-o', 'output_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Path to the ENA-derived Reads TSV file."
)
def main(
        assembly_tsv_path: Path,
        output_dir: Path
):
    assembly_df = pd.read_csv(assembly_tsv_path, sep='\t').rename(
        columns={'accession': 'assembly_accession'}
    )
    assembly_df['Participant'] = assembly_df['sample_title'].apply(extract_patient_name)
    download_all_patients(assembly_df, output_dir)


def extract_patient_name(sample_name: str):
    tokens = sample_name.split('_')

    if len(tokens) > 1 and tokens[1].startswith('T'):  # is a twin
        return "{}_{}".format(tokens[0], tokens[1])
    else:
        return tokens[0]


def download_all_patients(assembly_df: pd.DataFrame, output_dir: Path):
    for participant, section in assembly_df.groupby('Participant'):
        print(f"Handling participant {participant}")
        download_for_patient(section, str(participant), output_dir)


def download_for_patient(assembly_df: pd.DataFrame, target_participant: str, output_dir: Path):
    target_download_dir = output_dir / target_participant / 'isolate_assemblies'
    target_download_dir.mkdir(exist_ok=True, parents=True)
    metadata_path = target_download_dir / 'metadata.tsv'

    metadata_df_entries = []
    for _, row in assembly_df.iterrows():
        assembly_acc = row['assembly_accession']
        genus, species = row['scientific_name'].split()
        sample_title = row['sample_title']

        sample_title = sample_title[len(target_participant):]
        tokens = sample_title.split('_')
        if len(tokens) == 2:
            timepoint = tokens[1]
            sample_id = '*'
        elif len(tokens) == 3:
            timepoint = tokens[1]
            sample_id = tokens[2]
        else:
            raise ValueError(f"Was unable to parse the sample title `{sample_title}`.")

        fasta_path = target_download_dir / f'{assembly_acc}.fasta'
        print(f"Downloading assembly {assembly_acc} [{genus} {species}] -- timepoint {timepoint}")
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
    metadata_df = pd.DataFrame(metadata_df_entries)
    metadata_df.to_csv(metadata_path, sep='\t', index=False)


if __name__ == "__main__":
    main()
