"""
Take a representative genome, and introduce randomly sampled SNVs.
"""

from pathlib import Path
import click
import json
from typing import List, Dict


@click.command()
@click.option(
    '--input-json', '-i', 'input_json_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--output-json', '-o', 'output_json_path',
    type=click.Path(path_type=Path, dir_okay=False), required=True
)
@click.option(
    '--source-strains', '-s', 'source_strains',
    type=str, multiple=True, required=True
)
@click.option(
    '--mutant-strains', '-m', 'mutant_strains',
    type=str, multiple=True, required=True
)
@click.option(
    '--sim-genome-dir', '-g', 'sim_genome_dir',
    type=click.Path(path_type=Path, file_okay=False, exists=True, readable=True),
    required=True
)
@click.option(
    '--database-src-dir', '-ds', 'database_src_dir',
    type=click.Path(path_type=Path, file_okay=False, exists=True, readable=True),
    required=True
)
@click.option(
    '--database-tgt-dir', '-dt', 'database_tgt_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=True
)
def main(
        input_json_path: Path,
        output_json_path: Path,
        source_strains: List[str],
        mutant_strains: List[str],
        sim_genome_dir: Path,
        database_src_dir: Path,
        database_tgt_dir: Path
):
    if len(source_strains) != len(mutant_strains):
        print("Source strains' arglen must match mutant strains' arglen.")
        exit(1)

    src_to_mutant = {
        src_id: mutant_id
        for src_id, mutant_id in zip(source_strains, mutant_strains)
    }

    # ===== Create JSON file.
    print("Processing JSON database records.")
    with open(input_json_path, 'rt') as in_file:
        strain_entries = json.load(in_file)

    mutant_entries = []
    for strain_entry in strain_entries:
        if strain_entry['id'] in src_to_mutant:
            mutant_id = src_to_mutant[strain_entry['id']]
            mutant_entry = create_mutant_entry(strain_entry, mutant_id)
            mutant_entries.append(mutant_entry)

    output_json_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_json_path, 'w') as out_f:
        json.dump(strain_entries + mutant_entries, out_f, indent=4)

    # ==== Symlink genome FASTA files.
    print("Creating symlinks to genome files.")
    database_tgt_dir.mkdir(exist_ok=True, parents=True)
    for strain_entry in strain_entries:
        strain_id = strain_entry['id']
        for seq_entry in strain_entry['seqs']:
            seq_id = seq_entry['accession']
            fasta_path = database_src_dir / 'assemblies' / strain_id / f'{seq_id}.fasta'
            if not fasta_path.exists():
                print("Fasta sequence {} for strain {} does not exist!".format(fasta_path, strain_id))
                exit(1)

            target_path = database_tgt_dir / 'assemblies' / strain_id / f'{seq_id}.fasta'
            target_path.parent.mkdir(exist_ok=True, parents=True)
            if not target_path.exists():
                target_path.symlink_to(fasta_path)

    for m in mutant_strains:
        fasta_path = sim_genome_dir / f'{m}.fasta'
        target_path = database_tgt_dir / 'assemblies' / m / f'{m}.fasta'
        if target_path.exists():
            target_path.unlink()
        target_path.parent.mkdir(exist_ok=True, parents=True)
        target_path.symlink_to(fasta_path)


def create_mutant_entry(strain_entry: Dict, mutant_id: str):
    return {
        'id': mutant_id,
        'genus': strain_entry['genus'],
        'species': strain_entry['species'],
        'name': "{}.sim_mutant".format(strain_entry['name']),
        'genome_length': strain_entry['genome_length'],
        'seqs': [
            {
                'accession': mutant_id,
                'seq_type': 'chromosome'
            }
        ],
        'markers': [
            {
                'id': "{}.sim_mutant".format(marker_entry['id']),
                'name': marker_entry['name'],
                'type': marker_entry['type'],
                'source': mutant_id,
                'start': marker_entry['start'],
                'end': marker_entry['end'],
                'strand': marker_entry['strand'],
                'canonical': marker_entry['canonical']
            }
            for marker_entry in strain_entry['markers']
        ]
    }


if __name__ == "__main__":
    main()
