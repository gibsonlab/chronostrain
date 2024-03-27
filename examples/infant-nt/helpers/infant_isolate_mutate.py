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
    type=click.Path(path_type=Path, file_okay=False),
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
    index_df = pd.read_csv(src_isolate_idx, sep='\t')
    rng = np.random.default_rng(seed)

    path_mapping = {}
    fasta_output_dir.mkdir(exist_ok=True, parents=True)
    for _, row in index_df.loc[
        (index_df['Genus'] == 'Enterococcus')
        & (index_df['Species'] == 'faecalis')
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
