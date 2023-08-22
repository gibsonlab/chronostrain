"""
Take a representative genome, and introduce randomly sampled SNVs.
"""
import json
from pathlib import Path
from typing import Dict, Tuple, List

import click
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

base_alphabet = ['A', 'C', 'G', 'T']


@click.command()
@click.option(
    '--input-genome', '-i', 'input_genome_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--output-genome', '-o', 'output_genome_path',
    type=click.Path(path_type=Path, dir_okay=False), required=True
)
@click.option(
    '--json-db', '-j', 'json_chronostrain_db',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--density-snv', '-d', 'snv_density',
    type=float,
    help='A float between 0.0 and 1.0 indicating the density of SNVs. '
         'Each base will be mutated independently via a biased coin flip.'
)
@click.option(
    '--source-id', '-sid', 'source_id',
    type=str, required=True,
    help='The source accession to look for in the JSON file.'
)
@click.option(
    '--target-id', '-tid', 'target_id',
    type=str, required=False, default='',
    help='The desired target ID. '
         'For example, if the original fasta record\'s ID is an accession number (e.g. \"NZ_CP9058172.1\"), '
         'then a new human-readable ID might be \"NZ_CP9058172.1.sim_mutant\". '
         'If not specified, the script follows this convention.'
)
@click.option(
    '--seed', '-s', 'seed',
    type=int, required=True,
    help='The random seed to use for simulation. Required for reproducibility.'
)
def main(
        input_genome_path: Path,
        output_genome_path: Path,
        json_chronostrain_db: Path,
        snv_density: float,
        source_id: str,
        target_id: str,
        seed: int
):
    if snv_density < 0.0 or snv_density > 1.0:
        print("snv density must be between 0.0 and 1.0.")
        exit(1)

    record = SeqIO.read(input_genome_path, "fasta")
    rng = np.random.default_rng(seed)

    strain_entry = search_json(json_chronostrain_db, source_id)

    marker_regions = [
        (m['start'], m['end'])
        for m in strain_entry['markers']
    ]
    mutated_seq = mutate(record.seq, snv_density, rng, markers=marker_regions)

    if len(target_id) == 0:
        target_id = f'{record.id}.sim_mutant'
    SeqIO.write(
        [SeqRecord(mutated_seq, id=target_id, description=f'Simulated mutant of {record.id}')],
        output_genome_path,
        'fasta'
    )


def search_json(json_chronostrain_db: Path, source_id: str) -> Dict:
    with open(json_chronostrain_db, "rt") as f:
        strain_entries = json.load(f)
        for s in strain_entries:
            if s['id'] == source_id:
                return s
        raise ValueError("Strain entry `{}` not found in {}.".format(
            source_id, json_chronostrain_db
        ))


def mutate(genome: Seq, density: float, rng: np.random.Generator, markers: List[Tuple[int, int]] = None) -> Seq:
    buf = list(str(genome))
    if markers is not None:
        mask = np.zeros(len(buf), dtype=bool)
        for start, end in markers:
            assert start < end
            mask[start-1:end] = True
        print("Found {} marker regions, spanning {} / {} bases. (ratio={:.7f})".format(
            len(markers),
            np.sum(mask),
            len(mask),
            np.sum(mask) / len(mask)
        ))
    else:
        mask = np.ones(len(buf), dtype=bool)
    rng_coins = rng.uniform(low=0, high=1.0, size=len(buf)) < density
    rng_coins = rng_coins & mask

    print(
        "Mutating {} bases out of {} (rate={})".format(
            np.sum(rng_coins), len(rng_coins),
            np.sum(rng_coins) / np.sum(mask)
        )
    )

    return Seq(''.join([
        mutate_base(base, rng) if coin else base
        for base, coin in zip(buf, rng_coins)
    ]))


def mutate_base(base: str, rng: np.random.Generator) -> str:
    remaining = [b for b in base_alphabet if b != base]
    k = rng.choice(len(remaining), size=1).item()
    return remaining[k]


if __name__ == "__main__":
    main()





