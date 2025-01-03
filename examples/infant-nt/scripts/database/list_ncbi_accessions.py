from pathlib import Path
import json
import click


@click.command()
@click.option(
    '--chronostrain-json-path', '-c', 'chronostrain_json_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    required=True,
    help="The source ChronoStrain JSON File."
)
def main(chronostrain_json_path: Path):
    with open(chronostrain_json_path, "rt") as f:
        db_records = json.load(f)

    for record in db_records:
        strain_id = record['id']
        strain_name = record['name']
        genus = record['genus']
        species = record['species']
        chrom_len = record['genome_length']

        seqs = record['seqs']
        assert len(seqs) == 1
        assembly_acc = seqs[0]['seq_path'].split("/")[-2]

        is_efaec = (genus == "Enterococcus") and (species == "faecalis")
        """
        strain_id strain_name assembly_acc genus species chrom_len
        """
        if not is_efaec:
            print(f"{strain_id},{strain_name},{assembly_acc},{genus},{species},{chrom_len}")


if __name__ == "__main__":
    main()
