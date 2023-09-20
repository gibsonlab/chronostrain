#!/usr/bin/env python3
"""
Prepare symlinks to chromosome files for strainGE database construction.
"""
import click
from pathlib import Path
import pandas as pd


@click.command()
@click.option(
    '--index-refseq', '-i', 'index_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="The TSV path indexing the refseq catalog."
)
@click.option(
    '--target-dir', '-t', 'target_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="The directory to which the symlinks will be created."
)
def main(index_path: Path, target_dir: Path):
    index_df = pd.read_csv(index_path, sep='\t')
    target_dir.mkdir(exist_ok=True, parents=True)
    for _, row in index_df.iterrows():
        genus = row['Genus']
        if genus != 'Escherichia' and genus != 'Shigella':
            continue

        acc = row['Accession']
        src_path = Path(row['SeqPath'])
        target_path = target_dir / f'{acc}.fasta'

        if target_path.exists():
            target_path.unlink()
        target_path.symlink_to(src_path)


if __name__ == "__main__":
    main()
