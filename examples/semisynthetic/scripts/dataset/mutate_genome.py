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
    help='A float between 0.0 and 1.0 indicating the density of SNVs on non-marker regions only. '
         'Each base will be mutated independently via a biased coin flip.'
)
@click.option(
    '--density-snv-marker', '-dm', 'marker_snv_density',
    type=float,
    help='A float between 0.0 and 1.0 indicating the density of SNVs across the marker regions only. '
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
        marker_snv_density: float,
        source_id: str,
        target_id: str,
        seed: int,
):
    if snv_density < 0.0 or snv_density > 1.0:
        print("SNV density must be between 0.0 and 1.0.")
        exit(1)
    if marker_snv_density < 0.0 or marker_snv_density > 1.0:
        print("marker SNV density must be between 0.0 and 1.0.")
        exit(1)

    record = SeqIO.read(input_genome_path, "fasta")
    rng = np.random.default_rng(seed)

    strain_entry = search_json(json_chronostrain_db, source_id)

    mutated_seq = mutate(
        genome=record.seq,
        density=snv_density,
        marker_specific_density=marker_snv_density,
        markers=[(m['start'], m['end']) for m in strain_entry['markers']],
        rng=rng
    )

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


def mutate(
        genome: Seq,
        density: float,
        marker_specific_density: float,
        markers: List[Tuple[int, int]],
        rng: np.random.Generator
) -> Seq:
    buf = list(str(genome))

    # Identify markers.
    mask = np.zeros(len(buf), dtype=bool)
    for start, end in markers:
        assert start < end
        mask[start-1:end] = True
    print("Found {} marker regions, spanning {} / {} bases. (markers / genome ratio = {:.7f})".format(
        len(markers),
        np.sum(mask),
        len(mask),
        np.sum(mask) / len(mask)
    ))

    # Generate RNG coins.
    genome_rng_coins = rng.uniform(low=0, high=1.0, size=len(buf)) < density
    genome_rng_coins = genome_rng_coins & (~mask)

    marker_rng_coins = rng.uniform(low=0, high=1.0, size=len(buf)) < marker_specific_density
    marker_rng_coins = marker_rng_coins & mask

    def print_summary(_coins, _mask, prefix=""):
        print(
            "\t{}: Mutating {} bases out of {} (rate={})".format(
                prefix,
                np.sum(_coins),
                np.sum(_mask),
                np.sum(_coins) / np.sum(_mask)
            )
        )
    print_summary(genome_rng_coins, ~mask, "Non-Markers")
    print_summary(marker_rng_coins, mask, "Markers")

    buf = [
        mutate_base(base, rng) if coin else base
        for base, coin in
        zip(buf, genome_rng_coins)
    ]  # mutate non-markers
    buf = [
        mutate_base(base, rng) if coin else base
        for base, coin in
        zip(buf, marker_rng_coins)
    ]  # mutate markers only

    return Seq(''.join(buf))


def mutate_base(base: str, rng: np.random.Generator) -> str:
    remaining = [b for b in base_alphabet if b != base]
    k = rng.choice(len(remaining), size=1).item()
    return remaining[k]


if __name__ == "__main__":
    main()





