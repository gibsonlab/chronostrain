from typing import Optional, Dict, Any

import click
from logging import Logger
from pathlib import Path
import json

from ..base import option


@click.command()
@click.pass_context
@option(
    '--marker-seeds', '-m', 'marker_seeds_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to a TSV file of marker seeds. Will parse the first two columns. "
         "column 1 is the marker seed name (e.g. gene name), "
         "and column 2 is the single-record FASTA path for the seed's nucleotide sequence."
)
@option(
    '--references', '-r', 'ref_index_path',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True),
    required=True,
    help="Path to a TSV file of target references. "
         "Must contain at least these seven columns: `Accession`, `Genus`, `Species`, "
         "`Strain`, `ChromosomeLen`, `SeqPath`, `GFF` "
         "(note: ChromosomeLen is currently only used as metadata, but is still required.)"
)
@option(
    '--blastdb', '-b', 'blast_db_name', type=str, required=True,
    help="The name of the BLAST database to use. IDs of this database must be contained in the reference index."
)
@option(
    '--blastdb-dir', '-bd', 'blast_db_dir',
    type=click.Path(path_type=Path, file_okay=False),
    required=False, default=None,
    help="If specified, will specify the directory for the BLAST database. Has the same effect as "
         "exporting the `BLASTDB` variable."
)
@option(
    '--out-path', '-o', 'json_output_path',
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    required=True,
    help="The target path to output the database into (JSON format, so .json extension is suggested)"
)
@option(
    '--min-pct-idty', 'min_pct_idty',
    type=int, required=False, default=75,
    help="The minimum percent identity cutoff to use for BLAST (blastn's -perc_identity argument)"
)
@option(
    '--ident-threshold', '-t', 'identity_threshold',
    type=float, required=False, default=0.002,
    help="The distance threshold to use for Agglomerative clustering (a fraction between 0 and 1), representing"
         "one minus the percent identity (converted to decimals) in the concatenated multiple alignment. "
         "Default is 0.002, which represents 99.8% sequence similarity."
)
@option(
    '--min-marker-len', 'min_marker_len',
    type=int, required=False, default=150,
    help="The minimum length for a BLAST hit to be considered a marker sequence. "
         "The default is set to 150, which is a typical Illumina read length. "
         "Recommendation is to set this value to the length of the shortest read that you expect, "
         "assuming this is known. Might cause unexpected behavior for longer reads "
         "(to be addressed in a future version.)"
)
@option(
    "--skip-symlink", "skip_symlink",
    is_flag=True, default=False,
    help="If specified, then skips a pre-processing step which uses the reference index (--references) and populates "
         "ChronoStrain's DATA_DIR with the specified assembled sequence FASTA files."
)
@option(
    "--skip-prune", "skip_prune",
    is_flag=True, default=False,
    help="If specified, skip the pruning step."
)
def main(
        ctx: click.Context,
        marker_seeds_path: Path,
        ref_index_path: Path,
        blast_db_name: str,
        blast_db_dir: Optional[Path],
        json_output_path: Path,
        min_pct_idty: int,
        identity_threshold: float,
        min_marker_len: int,
        skip_symlink: bool,
        skip_prune: bool
):
    """
    Perform posterior estimation using ADVI.
    """
    ctx.ensure_object(Logger)
    logger = ctx.obj

    from .helpers import create_chronostrain_db, marker_concatenated_multiple_alignments
    import pandas as pd
    from typing import Dict
    from chronostrain.database import JSONParser
    from chronostrain.config import cfg
    reference_df = pd.read_csv(ref_index_path, sep="\t")
    json_output_path.parent.mkdir(exist_ok=True, parents=True)

    # ============== Optional: preprocess reference_df into
    if skip_symlink:
        pass
    else:
        from chronostrain.database.parser.marker_sources import EntrezMarkerSource
        from chronostrain.util.entrez import fasta_filename

        logger.info(f"Creating symbolic links to reference catalog (target dir: {cfg.database_cfg.data_dir})")
        for _, row in reference_df.iterrows():
            strain_id = row['Accession']
            seq_path = Path(row['SeqPath'])
            if not seq_path.exists():
                raise FileNotFoundError("Reference index pointed to `{seq_path}`, but it does not exist.")
            target_dir = EntrezMarkerSource.assembly_subdir(cfg.database_cfg.data_dir, strain_id)
            target_dir.mkdir(exist_ok=True, parents=True)

            target_path = fasta_filename(strain_id, target_dir)
            if target_path.is_symlink():
                target_path.unlink()
            elif target_path.exists():
                logger.info(f"File {target_path} already exists. Skipping symlink.")
            target_path.symlink_to(seq_path)

    # ============== Step 1: initialize using BLAST.
    logger.info("Building raw DB using BLAST.")
    blast_result_dir = json_output_path.parent / f"_BLAST_{json_output_path.stem}"

    gene_paths: Dict[str, Path] = {}
    with open(marker_seeds_path) as seed_file:
        for line in seed_file:
            tokens = line.strip().split('\t')
            gene_name = tokens[0]
            gene_fasta_path = Path(tokens[1])
            if not gene_fasta_path.exists():
                raise FileNotFoundError(
                    f"Sequence file for marker `{gene_name}` does not exist (got: {gene_fasta_path})"
                )
            gene_paths[gene_name] = gene_fasta_path

    strain_entries = create_chronostrain_db(
        blast_result_dir=blast_result_dir,
        strain_df=reference_df,
        gene_paths=gene_paths,
        blast_db_dir=blast_db_dir,
        blast_db_name=blast_db_name,
        min_pct_idty=min_pct_idty,
        min_marker_len=min_marker_len,
        logger=logger
    )

    raw_json_path = json_output_path.with_stem(f'{json_output_path.stem}-1raw')
    with open(raw_json_path, 'w') as outfile:
        json.dump(strain_entries, outfile, indent=4)
        logger.info(f"Wrote raw blast DB entries to {raw_json_path}.")

    # ==== Initialize database instance.
    logger.info(f"Loading DB instance, using data directory: {cfg.database_cfg.data_dir}")
    raw_db = JSONParser(
        entries_file=raw_json_path,
        data_dir=cfg.database_cfg.data_dir,
        marker_max_len=cfg.database_cfg.parser_kwargs['marker_max_len'],
        force_refresh=False
    ).parse()

    pruned_json_path = json_output_path.with_stem(f'{json_output_path.stem}-2pruned')
    if not skip_prune:
        # ============== Step 2: prune using multiple alignments.
        logger.info("Pruning database by constructing multiple alignments.")
        logger.debug(f"Target: {pruned_json_path}")

        align_path = json_output_path.parent / f"_ALIGN_{json_output_path.stem}" / "multiple_alignment.fasta"
        marker_names = set(gene_paths.keys())
        marker_concatenated_multiple_alignments(raw_db, align_path, sorted(marker_names), logger)
        prune_json_db(raw_json_path, pruned_json_path, align_path, logger, identity_threshold)
    else:
        logger.info("Skipping pruning.")
        import shutil
        shutil.copy(raw_json_path, pruned_json_path)

    # ============== Step 3: check for overlaps.
    logger.info("Resolving overlaps.")
    from chronostrain.cli.commands.make_db.helpers.resolve_overlaps import find_and_resolve_overlaps
    with open(pruned_json_path, "r") as f:
        db_json = json.load(f)
    for strain in db_json:
        find_and_resolve_overlaps(strain, reference_df, logger)
    with open(json_output_path, 'w') as o:
        json.dump(db_json, o, indent=4)
    logger.info("Finished.")


def prune_json_db(raw_json_path: Path,
                  tgt_json_path: Path,
                  align_path: Path, logger: Logger,
                  identity_threshold: float):
    # parse json entries.
    entries: Dict[str, Dict[str, Any]] = {}
    strain_ids = []
    with open(raw_json_path, "r") as f:
        _initial_strain_entries = json.load(f)
        for strain_entry in _initial_strain_entries:
            accession = strain_entry['id']
            entries[accession] = strain_entry
            strain_ids.append(accession)

    # perform clustering.
    from .helpers import cluster_db
    clusters, cluster_reps = cluster_db(strain_ids, align_path, logger, ident_fraction=identity_threshold)

    # Create the clustered json.
    result_entries = []
    for cluster, rep in zip(clusters, cluster_reps):
        rep_strain_idx = cluster[rep]
        rep_strain = strain_ids[rep_strain_idx]

        cluster_entry = entries[rep_strain]
        cluster_entry['cluster'] = [
            "{}({})".format(strain_ids[s_idx], entries[strain_ids[s_idx]]['name'])
            for s_idx in cluster
        ]
        result_entries.append(cluster_entry)

    with open(tgt_json_path, 'w') as outfile:
        json.dump(result_entries, outfile, indent=4)

    logger.info("Before clustering: {} strains".format(len(entries)))
    logger.info("After clustering: {} strains".format(len(result_entries)))


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    my_logger = create_logger("chronostrain.make_db")
    try:
        main(obj=my_logger)
    except Exception as e:
        my_logger.exception(e)
        exit(1)
