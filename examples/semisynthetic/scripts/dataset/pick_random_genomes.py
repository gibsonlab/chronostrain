from pathlib import Path
import json
from typing import *

import click

import numpy as np
from numpy.random import Generator
import pandas as pd


def load_poppunk_cluster(cluster_csv: Path) -> pd.DataFrame:
    def fix_accession(t_id):
        tokens = t_id.split('_')
        return '{}.{}'.format('_'.join(tokens[:-1]), tokens[2])

    df = pd.read_csv(cluster_csv)
    df['Accession'] = df['Taxon'].map(fix_accession)
    return df.drop(columns=['Taxon'])


def load_chronostrain_cluster(chronostrain_json: Path) -> pd.DataFrame:
    with open(chronostrain_json, "rt") as f:
        strain_entries = json.load(f)

    df_entries = []
    for s in strain_entries:
        rep_id = s['id']
        members = [x.split('(')[0] for x in s['cluster']]
        for member in members:
            df_entries.append({
                'Accession': member,
                'Cluster': rep_id
            })
    return pd.DataFrame(df_entries)


def load_strainge_cluster() -> pd.DataFrame:
    p = Path("/mnt/e/semisynthetic_data/databases/straingst/clusters.tsv")
    df_entries = []
    with open(p, "rt") as f:
        for line in f:
            clust = line.strip().split('\t')
            rep = clust[0]
            for member in clust:
                df_entries.append({
                    'Accession': member,
                    'Cluster': rep
                })
    return pd.DataFrame(df_entries)


def parse_phylogroups(phylogroup_path: Path):
    df_entries = []
    with open(phylogroup_path, 'rt') as f:
        for line in f:
            tokens = line.strip().split('\t')
            accession = Path(tokens[0]).with_suffix('').with_suffix('').name
            phylogroup = tokens[4]
            df_entries.append({'Accession': accession, 'Phylogroup': phylogroup})
    return pd.DataFrame(df_entries)


def generate_cluster_df(index_df: pd.DataFrame, chronostrain_json: Path, poppunk_cluster_csv: Path, phylogroup_path: Path):
    return index_df[['Accession', 'Strain', 'Genus', 'Species']].merge(
        load_chronostrain_cluster(chronostrain_json),
        on='Accession', how='inner'
    ).rename(columns={'Cluster': 'ClusterChronostrain'}).merge(
        load_poppunk_cluster(poppunk_cluster_csv),
        on='Accession', how='inner'
    ).rename(columns={'Cluster': 'ClusterPopPUNK'}).merge(
        parse_phylogroups(phylogroup_path), on='Accession'
    ).fillna('?')


def sample_random(cluster_df: pd.DataFrame, size: int, rng: Generator) -> List[str]:
    samples = None
    while samples is None:
        selection = cluster_df.loc[cluster_df['Phylogroup'] == 'A'].sample(size, random_state=rng)
        n_unique_chronostrain_clusters = len(pd.unique(selection['ClusterChronostrain']))
        n_unique_poppunk_clusters = len(pd.unique(selection['ClusterPopPUNK']))
        if n_unique_poppunk_clusters == size and n_unique_chronostrain_clusters == size:
            samples = list(selection['ClusterChronostrain'])
    return samples


@click.command()
@click.option(
    '--index-path', '-i', 'index_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True,
)
@click.option(
    '--poppunk-clusters', '-p', 'poppunk_csv',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True,
)
@click.option(
    '--chronostrain-json', '-c', 'chronostrain_json',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True,
)
@click.option(
    '--phylogroups', '-ph', 'phylogroup_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True,
)
@click.option('--num-genomes', '-n', 'num_genomes', type=int, required=True)
@click.option('--seed', '-s', 'seed', type=int, required=True)
@click.option(
    '--abundance-csv', '-a', 'abundance_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--out', '-o', 'out_dir',
    type=click.Path(path_type=Path, file_okay=False), required=True
)
def main(index_path: Path, poppunk_csv: Path, chronostrain_json: Path, phylogroup_path: Path,
         num_genomes: int, seed: int,
         abundance_path: Path,
         out_dir: Path):
    index_df = pd.read_csv(index_path, sep='\t')
    cluster_df = generate_cluster_df(index_df, chronostrain_json, poppunk_csv, phylogroup_path)
    rng = np.random.default_rng(seed)
    print("Sampling random strain IDs.")
    random_strain_ids = sample_random(cluster_df, size=num_genomes, rng=rng)

    out_dir.mkdir(exist_ok=True, parents=True)
    acc_list_path = out_dir / 'target_genomes.txt'
    with open(acc_list_path, 'wt') as f:
        print("## SEED={}".format(seed), file=f)
        for x in random_strain_ids:
            print(x, file=f)

    for x in random_strain_ids:
        src_seq_path = Path(index_df.loc[index_df['Accession'] == x, 'SeqPath'].item())
        tgt_seq_path = out_dir / f'{x}.fasta'
        tgt_seq_path.unlink(missing_ok=True)
        tgt_seq_path.symlink_to(src_seq_path)

    tgt_abundance_path = out_dir / 'abundances.txt'
    with open(tgt_abundance_path, "wt") as out_f, open(abundance_path, "rt") as template_f:
        template_f.readline()
        print(",".join(
            ["T"] + [f'{x}.READSIM_MUTANT' for x in random_strain_ids]
        ), file=out_f)
        for line in template_f:
            print(line.strip(), file=out_f)


if __name__ == "__main__":
    main()

