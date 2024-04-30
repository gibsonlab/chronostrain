from typing import Dict, List
from pathlib import Path
import click


def generate_themisto_build_input(sequence_files: Dict[str, Path], clustering: Dict[str, str], sim_genome_paths: List[Path], seq_path: Path, clust_path: Path):
    with open(seq_path, 'w') as seq_file, open(clust_path, 'w') as clust_file:
        for acc, seq_path in sequence_files.items():
            if acc not in clustering:
                raise Exception(
                    f"Unable to find {acc} in clustering file. "
                    "This is unexpected behavior, since `--seq-input/-i` is supposed to be the input for poppunk."
                )

            print(seq_path, file=seq_file)
            print(clustering[acc], file=clust_file)

        for sim_idx, sim_genome in enumerate(sim_genome_paths):
            sim_cluster = f'SIM_{sim_idx}'
            print(sim_genome, file=seq_file)
            print(sim_cluster, file=clust_file)


def fix_accession(acc_to_fix):
    """
    this function is necessary because poppunk converts periods (.) into underscores, even when the accession is
    something like "LZ_0189358.1" and turns it into "LZ_0189358_1"
    """
    tokens = acc_to_fix.split("_")
    prefix = "_".join(tokens[:-1])
    suffix = tokens[-1]
    if len(suffix) > 1:
        return acc_to_fix
    else:
        return f'{prefix}.{suffix}'


@click.command()
@click.option(
    '--seq-input', '-i', 'seq_input',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--sim-genome', '-g', 'sim_genome_paths',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    multiple=True, required=False
)
@click.option(
    '--outdir', '-o', 'out_dir',
    type=click.Path(path_type=Path, dir_okay=True), required=True
)
@click.option(
    '--clustering-poppunk', '-c', 'clustering_poppunk',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
def main(seq_input: Path, sim_genome_paths: List[Path], out_dir: Path, clustering_poppunk: Path):
    if len(sim_genome_paths) == 0:
        print("No simulated genomes to add.")

    for p in sim_genome_paths:
        if not p.exists():
            print("Simulated genome file {} does not exist.".format(p))
            exit(1)

    # Load sequence input paths
    print(f"Reading sequence paths from {seq_input}")
    with open(seq_input, "rt") as f:
        seq_files = {}
        for line in f:
            s_id, s_path = line.strip().split('\t')
            seq_files[s_id] = Path(s_path)

    # Load clusters
    print(f"Reading clusters from {clustering_poppunk}")
    with open(clustering_poppunk, "rt") as f:
        clustering = {}
        for line in f:
            taxon, cluster_id = line.strip().split(',')
            if taxon == "Taxon":
                continue
            taxon = fix_accession(taxon)
            clustering[taxon] = cluster_id

    out_dir.mkdir(exist_ok=True, parents=True)
    seq_path = out_dir / 'sequences.txt'
    clust_path = out_dir / 'clusters.txt'
    generate_themisto_build_input(seq_files, clustering, sim_genome_paths, seq_path, clust_path)
    print("Wrote themisto build input files.")


if __name__ == "__main__":
    main()
