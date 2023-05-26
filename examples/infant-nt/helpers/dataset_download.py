from pathlib import Path
import pandas as pd

from urllib.parse import urlparse
from ftplib import FTP

import click


def download_from_ftp(ftp_url: str, target_path: Path):
    if ftp_url.startswith('ftp://'):
        url_parsed = urlparse(ftp_url)
    else:
        url_parsed = urlparse(f'ftp://{ftp_url}')

    print(f"Downloading resource {ftp_url}")

    ftp = FTP(url_parsed.netloc)
    ftp.login()
    with open(target_path, 'wb') as f:
        ftp.retrbinary(f'RETR {url_parsed.path}', f.write)


def download_fastq(target_url: str, target_path: Path):
    mark = target_path.with_suffix(".DONE")
    if mark.exists():
        print('File {} already exists; skipping download.'.format(target_path))
    else:
        download_from_ftp(
            target_url, target_path
        )
        mark.touch(exist_ok=True)


def download_all(dataset: pd.DataFrame, out_dir: Path, target_participant: str):
    df_entries = []
    for _, row in dataset.iterrows():
        sample_accession = row['sample_accession']
        sample_title = row['sample_title']

        if not sample_title.startswith(target_participant):
            continue

        tokens = sample_title.split('_')
        if len(tokens) == 2:
            participant, timepoint = tokens
            sample_id = "*"
        elif len(tokens) == 3:
            participant, timepoint, sample_id = tokens
        else:
            print(f"Unparsable sample title `{sample_title}`")
            continue

        try:
            timepoint = float(timepoint)
        except ValueError:
            print(f"Unparseable timepoint string `{timepoint}` (src={sample_title})")
            continue

        print(f"Fetching sample {sample_accession} "
              f"[participant {participant}, timepoint {timepoint}, sample_id {sample_id}]")
        out_dir.mkdir(exist_ok=True, parents=True)

        urls = row['fastq_ftp'].split(';')
        fastq1 = out_dir / '{}_1.fastq.gz'.format(sample_accession)
        fastq2 = out_dir / '{}_2.fastq.gz'.format(sample_accession)
        download_fastq(urls[0], fastq1)
        download_fastq(urls[1], fastq2)

        df_entries.append({
            "Participant": target_participant,
            "T": timepoint,
            "SampleId": sample_accession,
            "Read1": str(fastq1),
            "Read2": str(fastq2)
        })

    return pd.DataFrame(df_entries)


@click.command()
@click.option(
    '--metagenomic-tsv', '-m', 'metagenomic_tsv',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to the ENA-derived Project TSV file."
)
@click.option(
    '--out-dir', '-o', 'out_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="The target output directory."
)
@click.option(
    '--participant', '-p', 'target_participant',
    type=str,
    required=True,
    help="The participant ID."
)
def main(
        metagenomic_tsv: Path,
        out_dir: Path,
        target_participant: str
):
    target_dir = out_dir / target_participant
    dataset_df = download_all(
        pd.read_csv(metagenomic_tsv, sep='\t'),
        target_dir / "reads",
        target_participant
    )
    dataset_df.to_csv(target_dir / "dataset.tsv", sep='\t', index=False)


if __name__ == "__main__":
    main()
