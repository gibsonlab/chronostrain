import click
from pathlib import Path

from ..base import option


@click.command()
@option(
    '--input', '-i', 'source_json_path',
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    help="The JSON database file."
)
@option(
    '--output', '-o', 'output_cluster_path',
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    help="The file path to which the clusters will be output."
)
@option(
    '--ident-threshold', '-t', 'identity_threshold',
    type=float, required=False, default=0.998,
    help="The distance threshold to use for Agglomerative clustering (a fraction between 0 and 1), representing"
         "one minus the percent identity (converted to decimals) in the concatenated multiple alignment. "
         "Default is 0.998, which represents 99.8% sequence similarity."
)
def main(
        source_json_path: Path,
        output_cluster_path: Path,
        identity_threshold: float
):
    """
    Cluster a JSON database's strain entries using markers' estimated sequence similarity.
    """
    from chronostrain.logging import create_logger
    logger = create_logger("chronostrain.cli.prune_json")

    # ============== Step 3: prune using clustering on genomic distances.
    logger.info("Pruning database via clustering")
    logger.debug(f"Src: {source_json_path}, Output: {output_cluster_path}")
    logger.info("Target identity threshold = {}".format(identity_threshold))

    from chronostrain.config import cfg
    from .helpers import cluster_json_db_jaccard
    cluster_json_db_jaccard(
        src_json_path=source_json_path,
        output_path=output_cluster_path,
        cfg=cfg, logger=logger,
        tmp_dir=source_json_path.resolve().parent / '__cluster_tmp',
        identity_threshold=identity_threshold
    )


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    main_logger = create_logger("chronostrain.MAIN")
    try:
        main()
    except Exception as e:
        main_logger.exception(e)
        exit(1)
