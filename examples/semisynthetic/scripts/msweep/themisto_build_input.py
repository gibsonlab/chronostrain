from typing import Dict, List
from pathlib import Path
import pandas as pd
import click


def generate_themisto_build_input(index_path: Path, clustering: Dict[str, str], sim_genome_paths: List[Path], seq_path: Path, clust_path: Path):
    index_df = pd.read_csv(index_path, sep='\t')

    with open(seq_path, 'w') as seq_file, open(clust_path, 'w') as clust_file:
        for _, row in index_df.iterrows():
            acc = row['Accession']
            seq_path = row['SeqPath']

            if acc not in clustering:
                genus = row['Genus']
                species = row['Species']
                strain = row['Strain']
                print(f"[WARNING] Unable to find {acc} ({genus} {species}, strain {strain}) in clustering file. Is an entry missing?")
                continue

            print(seq_path, file=seq_file)
            print(clustering[acc], file=clust_file)

        for sim_idx, sim_genome in enumerate(sim_genome_paths):
            sim_cluster = f'SIM_{sim_idx}'
            print(sim_genome, file=seq_file)
            print(sim_cluster, file=clust_file)


def fix_accession(broken_acc):
    # this function is necessary because poppunk converts periods (.) into underscores.
    tokens = broken_acc.split("_")
    prefix = "_".join(tokens[:-1])
    suffix = tokens[-1]
    return f'{prefix}.{suffix}'


@click.command()
@click.option(
    '--refseq-index', '-i', 'refseq_index',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--sim-genome', '-g', 'sim_genome_paths',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    multiple=True, required=True
)
@click.option(
    '--outdir', '-o', 'out_dir',
    type=click.Path(path_type=Path, dir_okay=True), required=True
)
@click.option(
    '--clustering-poppunk', '-c', 'clustering_poppunk',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
def main(refseq_index: Path, sim_genome_paths: List[Path], out_dir: Path, clustering_poppunk: Path):
    if len(sim_genome_paths) == 0:
        print("No simulated genomes to add.")
        exit(1)
    for p in sim_genome_paths:
        if not p.exists():
            print("Simulated genome file {} does not exist.".format(p))
            exit(1)

    out_dir.mkdir(exist_ok=True, parents=True)
    with open(clustering_poppunk, "rt") as f:
        clustering = {}
        for line in f:
            taxon, cluster_id = line.strip().split(',')
            if taxon == "Taxon":
                continue
            taxon = fix_accession(taxon)
            clustering[taxon] = cluster_id

    seq_path = out_dir / 'sequences.txt'
    clust_path = out_dir / 'clusters.txt'
    generate_themisto_build_input(refseq_index, clustering, sim_genome_paths, seq_path, clust_path)
    print("Wrote themisto build input files.")


if __name__ == "__main__":
    main()
