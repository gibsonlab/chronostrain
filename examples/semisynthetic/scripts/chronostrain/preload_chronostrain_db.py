import click
from pathlib import Path
from chronostrain.config import cfg
from chronostrain.model import StrainCollection
from chronostrain.util.external import bowtie2_build


@click.command()
@click.option(
    '--strain-subset', '-s', 'strain_subset_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=False, readable=True),
    required=False, default=None,
    help="A text file specifying a subset of database strain IDs to perform filtering with; "
         "a TSV file containing one ID per line, optionally with a second column for metadata.",
)
def main(strain_subset_path: Path):
    db = cfg.database_cfg.get_database()
    if strain_subset_path is not None:
        with open(strain_subset_path, "rt") as f:
            strain_collection = StrainCollection(
                [db.get_strain(line.strip().split('\t')[0]) for line in f if not line.startswith("#")],
                db.signature
            )
    else:
        strain_collection = StrainCollection(db.all_strains(), db.signature)

    marker_reference_path = strain_collection.multifasta_file
    bowtie2_build(
        refs_in=[marker_reference_path],
        index_basepath=marker_reference_path.parent,
        index_basename=marker_reference_path.stem,
        offrate=1,  # default is 5; but we want to optimize for the -a option.
        ftabchars=13,
        quiet=True,
        threads=cfg.model_cfg.num_cores,
    )


if __name__ == "__main__":
    main()
