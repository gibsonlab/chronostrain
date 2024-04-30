from typing import  *
import click
from pathlib import Path
import pandas as pd
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


base_alphabet = ['A', 'C', 'G', 'T']


def mutate_base(base: str, rng: np.random.Generator) -> str:
    remaining = [b for b in base_alphabet if b != base]
    k = rng.choice(len(remaining), size=1).item()
    return remaining[k]


def mutate_seq(
        seq: Seq,
        density: float,
        rng: np.random.Generator
) -> Tuple[Seq, int]:
    buf = list(str(seq))

    # Generate RNG coins.
    genome_rng_coins = rng.uniform(low=0, high=1.0, size=len(buf)) < density

    buf = [
        mutate_base(base, rng) if coin else base
        for base, coin in
        zip(buf, genome_rng_coins)
    ]  # mutate non-markers

    return Seq(''.join(buf)), int(np.sum(genome_rng_coins))


def mutate_genome(src_fasta: Path, out_fasta: Path, mutation_rate: float, rng: np.random.Generator):
    # read a multi-fasta file
    out_records = []
    n_records = 0
    n_total_bases = 0
    n_total_mutations = 0
    for record in SeqIO.parse(src_fasta, "fasta"):
        mutated_seq, n_mutations = mutate_seq(record.seq, mutation_rate, rng)
        out_records.append(
            SeqRecord(
                seq=mutated_seq,
                id=record.id,
                name=record.name,
                description=f"mutation_p={mutation_rate}"
            )
        )
        n_total_bases += len(mutated_seq)
        n_total_mutations += n_mutations
        n_records += 1

    frac = n_total_mutations / n_total_bases
    print(f"Mutated {n_total_mutations} / {n_total_bases} nucleotides across {n_records} records (fraction={frac}).")
    with open(out_fasta, "wt") as out_f:
        SeqIO.write(out_records, out_f, "fasta")


@click.command()
@click.option(
    '--isolate-index', '-i', 'src_isolate_idx',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="The TSV path indexing the isolate catalog."
)
@click.option(
    '--out-index', '-o', 'tgt_isolate_idx',
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="The target output isolate catalog."
)
@click.option(
    '--mutation-rate', '-m', 'mutation_rate',
    type=float, required=True,
    help="The mutation rate for simulated variatn generation."
)
@click.option(
    '--dir', '-d', 'fasta_output_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="The directory to which the resulting FASTA files will be stored."
)
@click.option(
    '--seed', '-s', 'seed',
    type=int, required=True,
    help='The random seed to use for simulation. Required for reproducibility.'
)
def main(src_isolate_idx: Path, tgt_isolate_idx: Path, mutation_rate: float, fasta_output_dir: Path, seed: int):
    rng = np.random.default_rng(seed)

    # These are all infnat isolates that were positively identified in stool by mGEMS using chronostrain-mirror db.
    genomes_to_mutate = {'GCA_902161795', 'GCA_902163505', 'GCA_902163065', 'GCA_902161695', 'GCA_902165735', 'GCA_902159995', 'GCA_902161935', 'GCA_902159175', 'GCA_902161045', 'GCA_902160005', 'GCA_902163415', 'GCA_902163575', 'GCA_902163025', 'GCA_902163085', 'GCA_902163175', 'GCA_902158995', 'GCA_902163125', 'GCA_902163115', 'GCA_902159455', 'GCA_902160235', 'GCA_902165825', 'GCA_902161845', 'GCA_902159115', 'GCA_902158775', 'GCA_902159445', 'GCA_902163815', 'GCA_902163875', 'GCA_902162285', 'GCA_902162985', 'GCA_902160165', 'GCA_902162095', 'GCA_902161145', 'GCA_902159695', 'GCA_902164645', 'GCA_902163845', 'GCA_902162035', 'GCA_902159275', 'GCA_902161275', 'GCA_902161445', 'GCA_902161945', 'GCA_902158935', 'GCA_902164995', 'GCA_902161335', 'GCA_902161895', 'GCA_902162275', 'GCA_902163165', 'GCA_902164775', 'GCA_902163015', 'GCA_902163055', 'GCA_902166645', 'GCA_902164245', 'GCA_902165505', 'GCA_902162795', 'GCA_902159985', 'GCA_902161785', 'GCA_902163585', 'GCA_902162245', 'GCA_902163635', 'GCA_902162135', 'GCA_902163705', 'GCA_902159955', 'GCA_902162885', 'GCA_902165985', 'GCA_902158825', 'GCA_902160125', 'GCA_902161645', 'GCA_902162085', 'GCA_902159935', 'GCA_902163155', 'GCA_902162955', 'GCA_902163645', 'GCA_902158905', 'GCA_902162335', 'GCA_902159975', 'GCA_902164385', 'GCA_902162935', 'GCA_902159085', 'GCA_902163915', 'GCA_902160975', 'GCA_902165575', 'GCA_902160505', 'GCA_902158765', 'GCA_902159705', 'GCA_902165295', 'GCA_902163615', 'GCA_902159395', 'GCA_902158945', 'GCA_902158795', 'GCA_902159565', 'GCA_902164735', 'GCA_902158885', 'GCA_902161375', 'GCA_902165815', 'GCA_902165515', 'GCA_902158925', 'GCA_902164275', 'GCA_902159255', 'GCA_902164125', 'GCA_902159145', 'GCA_902161805', 'GCA_902164785', 'GCA_902162945', 'GCA_902164345', 'GCA_902158985', 'GCA_902164115', 'GCA_902161745', 'GCA_902158955', 'GCA_902163925', 'GCA_902163805', 'GCA_902163825', 'GCA_902160075', 'GCA_902165485', 'GCA_902158855', 'GCA_902158895', 'GCA_902162865', 'GCA_902163655', 'GCA_902166365'}
    print("Will mutate {} isolate genomes.".format(len(genomes_to_mutate)))

    index_df = pd.read_csv(src_isolate_idx, sep='\t')
    index_df = index_df.loc[index_df['Accession'].isin(genomes_to_mutate)]

    path_mapping = {}
    fasta_output_dir.mkdir(exist_ok=True, parents=True)
    for _, row in index_df.loc[
        (index_df['Genus'] == 'Enterococcus')
        & (index_df['Species'] == 'faecalis')
        & index_df['Accession'].isin(genomes_to_mutate)
    ].iterrows():
        acc = row['Accession']
        print(f"Handling {acc}")
        src_fasta = Path(row['SeqPath'])
        target_fasta = fasta_output_dir / f'{acc}.fasta'
        mutate_genome(src_fasta, target_fasta, mutation_rate, rng)
        path_mapping[acc] = str(target_fasta)

    index_df['SeqPath'] = index_df['Accession'].map(path_mapping)
    index_df.to_csv(tgt_isolate_idx, index=False, sep='\t')


if __name__ == "__main__":
    main()
