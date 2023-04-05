from pathlib import Path
import pandas as pd

from urllib.parse import urlparse
from ftplib import FTP

import click
from chronostrain import create_logger
logger = create_logger('script.dataset_download')



def download_all(dataset: pd.DataFrame, out_dir: Path):
    for _, row in dataset.iterrows():
        sample_accession = row['sample_accession']
        sample_title = row['sample_title']
        tokens = sample_title.split('_')
        if len(tokens) != 2:
            raise ValueError("Unparsable sample title `{}`".format(sample_title))

        participant, timepoint = tokens

        # ================== testing on one participant only, for development:
        if participant != '513122':
            continue

        logger.debug(f"Fetching sample {sample_accession} [participant {participant}, timepoint {timepoint}]")
        target_dir = out_dir / participant / 'reads'
        target_dir.mkdir(exist_ok=True, parents=True)

        urls = row['fastq_ftp'].split(';')
        fastq1 = target_dir / '{}_{}_1.fastq.gz'.format(timepoint, sample_accession)
        fastq2 = target_dir / '{}_{}_2.fastq.gz'.format(timepoint, sample_accession)
        download_fastq(urls[0], fastq1)
        download_fastq(urls[1], fastq2)


@click.command()
@click.option(
    '--project-tsv', '-p', 'project_tsv',
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
def main(
        project_tsv: Path,
        out_dir: Path
):
    download_all(
        pd.read_csv(project_tsv, sep='\t'),
        out_dir
    )


if __name__ == "__main__":
    main()
