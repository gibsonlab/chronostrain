from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import click
from contextlib import contextmanager

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from chronostrain.logging import create_logger
from chronostrain.util.external import art_illumina, call_command
logger = create_logger('msweep_threshold_calculator')


@dataclass
class GenomeInfo:
    accession: str
    seq_path: Path


@dataclass
class ThemistoConfig:
    db_dir: Path
    db_prefix: str
    n_threads: int


@dataclass
class MSweepConfig:
    cluster_file: Path
    n_threads: int


@dataclass
class ArtConfig:
    profile_1: Path
    profile_2: Path
    output_sam: bool
    output_aln: bool


def fix_accessions(poppunk_acc: str) -> str:
    tokens = poppunk_acc.split('_')
    suffix = tokens[-1]
    prefix = '_'.join(tokens[:-1])
    return f'{prefix}.{suffix}'


def pick_sequences_for_sampling(merged_clust_df: pd.DataFrame):
    # 1. exclude clusters that are the only clusters for a species.
    exclusion_set_1 = set()
    for species, species_section in merged_clust_df.groupby("Species"):
        species_clusters = pd.unique(species_section['Cluster'])
        if len(species_clusters) == 1:
            exclusion_set_1.add(species_clusters[0])

    logger.info("Excluding {} clusters, because they are singletons for their species.".format(len(exclusion_set_1)))
    merged_clust_df = merged_clust_df.loc[~merged_clust_df['Cluster'].isin(exclusion_set_1)]

    # 2. exclude singleton clusters.
    exclusion_set_2 = set()
    counts_df = merged_clust_df.groupby("Cluster")['Accession'].count().rename("ClusterSize").to_frame().reset_index()
    for _, row in counts_df.loc[counts_df['ClusterSize'] == 1].iterrows():
        clust = row['Cluster']
        exclusion_set_2.add(clust)

    logger.info("Excluding {} clusters, because they only contain 1 genome.".format(len(exclusion_set_2)))
    merged_clust_df = merged_clust_df.loc[~merged_clust_df['Cluster'].isin(exclusion_set_2)]

    logger.info("{} Clusters remaining.".format(
        len(pd.unique(merged_clust_df['Cluster']))
    ))

    # 3. Compute a random set of of sqrt(C) for each cluster C.
    genomes_for_sampling = {}
    for clust, clust_section in merged_clust_df.groupby("Cluster"):
        all_members = clust_section['Accession'].tolist()
        selection = list(np.random.choice(
            all_members,
            size=int(math.sqrt(len(all_members))),
            replace=False,
        ))
        genomes_for_sampling[clust] = [
            GenomeInfo(acc, Path(clust_section.loc[clust_section['Accession'] == acc, 'SeqPath'].item()))
            for acc in selection
        ]
    return exclusion_set_1, exclusion_set_2, genomes_for_sampling


def perform_simulation_thresholding(
        genomes_for_sampling: Dict[str, List[GenomeInfo]],
        n_replicates: int,
        n_reads_per_replicate: int,
        tmp_dir: Path,
        art_cfg: ArtConfig,
        themisto_cfg: ThemistoConfig,
        msweep_cfg: MSweepConfig
):
    sections = []

    seed = 22390
    for clust_idx, (clust, genomes) in enumerate(genomes_for_sampling.items()):  # Do this for each cluster.
        for i in tqdm(range(n_replicates), desc=f'CIDX {clust_idx} of {len(genomes_for_sampling)}'):
            # Pick a random genome.
            random_genome = np.random.choice(genomes, size=1)[0]
            with replicate_inference(
                    random_genome, tmp_dir, seed, n_reads_per_replicate, art_cfg, themisto_cfg, msweep_cfg
            ) as inference_result:
                sections.append(
                    inference_result.assign(
                        Source=clust,
                        Replicate=i
                    ).rename(columns={'Cluster': 'Target'})
                )
            seed += 1
            break  # debug
        break  # debug

    return pd.concat(sections, ignore_index=True)


@contextmanager
def replicate_inference(
        genome: GenomeInfo, work_dir: Path, seed: int, n_reads: int,
        art_cfg: ArtConfig, themisto_cfg: ThemistoConfig, msweep_cfg: MSweepConfig,
) -> pd.DataFrame:
    # 1. Simulate reads.
    logger.info("Simulating reads.")
    fwd_reads, rev_reads = art_illumina(
        reference_path=genome.seq_path,
        num_reads=n_reads,
        output_dir=work_dir,
        output_prefix=f'__{genome.accession}_sim_',
        profile_first=art_cfg.profile_1,
        profile_second=art_cfg.profile_2,
        read_length=150,
        seed=seed,
        output_sam=art_cfg.output_sam,
        output_aln=art_cfg.output_aln,
        silent=True
    )

    # 2. Perform pseudoalignment.
    themisto_input_list = work_dir / 'input_files.txt'
    themisto_output_list = work_dir / 'output_files.txt'
    fwd_aln_file = work_dir / 'fwd_alns.txt'
    rev_aln_file = work_dir / 'rev_alns.txt'
    with open(themisto_input_list, 'w') as input_list_f, open(themisto_output_list, 'w') as output_list_f:
        print(fwd_reads, file=input_list_f)
        print(fwd_aln_file, file=output_list_f)
        print(rev_reads, file=input_list_f)
        print(rev_aln_file, file=output_list_f)

    themisto_tmpdir = work_dir / '__TMP'
    logger.info("Running pseudoalignment.")
    call_command(
        'themisto',
        [
            'pseudoalign',
            '--query-file-list', themisto_input_list,
            '--out-file-list', themisto_output_list,
            '--index-prefix', themisto_cfg.db_prefix,
            '--temp-dir', themisto_tmpdir,
            '--n-threads', themisto_cfg.n_threads,
            '--sort-output'
        ],
        cwd=themisto_cfg.db_dir,
        silent=True
    )

    # 3. Perform algs.
    logger.info("Performing algs.")
    call_command(
        'mSWEEP',
        [
            '--themisto-1', fwd_aln_file,
            '--themisto-2', rev_aln_file,
            '-i', msweep_cfg.cluster_file,
            '-t', msweep_cfg.n_threads,
            '-o', 'algs'
        ],
        cwd=work_dir,
        silent=True
    )

    inference_output_file = work_dir / 'inference_abundances.txt'
    assert inference_output_file.exists()
    try:
        yield parse_msweep_output(inference_output_file)
    finally:
        themisto_input_list.unlink()  # clean up
        themisto_output_list.unlink()  # clean up
        fwd_reads.unlink()  # clean up
        rev_reads.unlink()  # clean up
        fwd_aln_file.unlink()  # clean up
        rev_aln_file.unlink()  # clean up
        inference_output_file.unlink()  # clean up
        themisto_tmpdir.rmdir()  # clean up


def parse_msweep_output(msweep_result: Path) -> pd.DataFrame:
    df_entries = []
    with open(msweep_result, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            clust_id, mean_ratio = line.strip().split('\t')
            df_entries.append({'Cluster': clust_id, 'Abund': float(mean_ratio)})
    return pd.DataFrame(df_entries)


def main(
        clust_path: Path,
        index_path: Path,
        out_path: Path,
        replicates_per_cluster: int,
        tmp_dir: Path,
        art_profile_1: Path,
        art_profile_2: Path,
        themisto_db_dir: Path,
        themisto_db_prefix: str,
        n_threads: int
):
    np.random.seed(31415)
    clust_df = pd.read_csv(clust_path)
    clust_df['Accession'] = clust_df['Taxon'].map(fix_accessions)

    index_df = pd.read_csv(index_path, sep='\t')

    merged = clust_df.merge(index_df, on='Accession', how='inner').drop(columns=['Taxon', 'GFF', 'Assembly', 'ChromosomeLen'])
    species_singletons, cluster_singletons, genomes_for_sampling = pick_sequences_for_sampling(merged)

    tmp_dir.mkdir(exist_ok=True, parents=True)
    simulation_results = perform_simulation_thresholding(
        genomes_for_sampling,
        n_replicates=replicates_per_cluster,
        n_reads_per_replicate=100000,
        tmp_dir=tmp_dir,
        art_cfg=ArtConfig(art_profile_1, art_profile_2, False, False),
        themisto_cfg=ThemistoConfig(themisto_db_dir, themisto_db_prefix, n_threads),
        msweep_cfg=MSweepConfig(themisto_db_dir / 'clusters.txt', n_threads)
    )
    tmp_dir.rmdir()

    simulation_results.to_feather(out_path)


if __name__ == "__main__":
    main(
        clust_path=Path("/mnt/e/semisynthetic_data/poppunk/refine/refine_clusters.csv"),
        index_path=Path("/mnt/e/ecoli_db/ref_genomes/index.tsv"),
        out_path=Path("/mnt/e/semisynthetic_data/msweep_thresholds/estimates.feather"),
        replicates_per_cluster=5,
        tmp_dir=Path("/mnt/e/semisynthetic_data/msweep_thresholds/__REPLICATES_TMP"),
        art_profile_1=Path("/home/youn/work/chronostrain/examples/semisynthetic/files/HiSeqReference"),
        art_profile_2=Path("/home/youn/work/chronostrain/examples/semisynthetic/files/HiSeqReference"),
        themisto_db_dir=Path("/mnt/e/semisynthetic_data/msweep_thresholds/pseudoalignment_index"),
        themisto_db_prefix='enterobacteriaceae',
        n_threads=8
    )
