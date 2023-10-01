from pathlib import Path
from logging import Logger
import click

from chronostrain.database.backend import PandasAssistedBackend
from chronostrain.model import StrainMetadata
from chronostrain.database import StrainDatabase, PickleParser
from ..base import option


@click.command()
@option(
    '--input-db', '-i', 'input_db_name',
    type=str, required=True, help="The target database file to prune (assumes a pickle exists)."
)
@option(
    '--output-db', '-o', 'output_db_name',
    type=str, required=True, help="The target database file to prune (assumes a pickle exists)."
)
@option(
    '--ident-threshold', '-t', 'identity_threshold',
    type=float, required=False, default=0.998,
    help="The distance threshold to use for Agglomerative clustering (a fraction between 0 and 1), representing"
         "one minus the percent identity (converted to decimals) in the concatenated multiple alignment. "
         "Default is 0.998, which represents 99.8% sequence similarity."
)
def main(
        input_db_name: str,
        output_db_name: str,
        identity_threshold: float,
):
    """
    Prune an existing pickle file using multiple alignment & clustering.
    """
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli.prune_pickle")

    from chronostrain.config import cfg

    # ==== Initialize database instance.
    logger.info(f"Loading DB instance, using data directory: {cfg.database_cfg.data_dir}")
    input_db = PickleParser(input_db_name, cfg.database_cfg.data_dir).parse()

    # ============== Step 2: prune using multiple alignments.
    logger.info("Pruning database by constructing multiple alignments.")
    from chronostrain.cli.commands.make_db.helpers.multiple_alignments import marker_concatenated_multiple_alignments

    marker_names = sorted(input_db.all_marker_names())
    align_path = input_db.work_dir / "multiple_alignment.fasta"
    marker_concatenated_multiple_alignments(input_db, align_path, marker_names, logger)
    prune_db(input_db, output_db_name, align_path, identity_threshold, logger, cfg)


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    my_logger = create_logger("chronostrain.prune_pickle")
    try:
        main(obj=my_logger)
    except Exception as e:
        my_logger.exception(e)
        exit(1)


def prune_db(input_db: StrainDatabase, output_db_name: str, alignments_path: Path, identity_threshold: float, logger: Logger, cfg):
    from .helpers import cluster_db
    strains = input_db.all_strains()
    clusters, cluster_reps, _ = cluster_db(
        strain_ids=[s.id for s in strains],
        strain_entries=[
            {'markers': [{'name': m.name} for m in s.markers]}
            for s in strains
        ],
        alignments_path=alignments_path,
        logger=logger,
        ident_fraction=identity_threshold
    )

    # Create the clustered database.
    new_strains = []
    for cluster, rep in zip(clusters, cluster_reps):
        rep_strain_idx = cluster[rep]
        rep_strain = strains[rep_strain_idx]
        if rep_strain.metadata is None:
            # noinspection PyTypeChecker
            rep_strain.metadata = StrainMetadata(
                None,
                None,
                None,
                None,
                None,
                [strains[s_idx].id for s_idx in cluster]
            )
        new_strains.append(rep_strain)

    new_db = StrainDatabase(
        backend=PandasAssistedBackend(),
        data_dir=cfg.database_cfg.data_dir,
        name=output_db_name,
    )
    serializer = PickleParser(output_db_name, cfg.database_cfg.data_dir)

    new_db.backend.add_strains(new_strains)
    new_db.save_markers_to_multifasta(force_refresh=True)
    serializer.save_to_disk(new_db)

    logger.info("Before clustering: {} strains".format(input_db.num_strains()))
    logger.info("After clustering: {} strains".format(new_db.num_strains()))
    logger.info("Successfully created new database instance `{}`.".format(output_db_name))
    logger.info("Pickle path: {}".format(serializer.disk_path()))
