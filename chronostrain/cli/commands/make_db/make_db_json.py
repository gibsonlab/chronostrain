import shutil
from typing import Optional

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
    type=float, required=False, default=0.998,
    help="The distance threshold to use for Agglomerative clustering (a fraction between 0 and 1), representing"
         "one minus the percent identity (converted to decimals) in the concatenated multiple alignment. "
         "Default is 0.998, which represents 99.8% sequence similarity."
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
    '--threads', 'num_threads',
    type=int, required=False, default=1,
    help="The number of threads to use (e.g. for blast)."
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
        num_threads: int,
        skip_symlink: bool,
        skip_prune: bool
):
    """
    Create a database using marker seeds.
    Requires BLAST, and as input takes an index TSV file of reference assemblies.
    """
    ctx.ensure_object(Logger)
    logger = ctx.obj

    from .helpers import create_chronostrain_db, prune_json_db_jaccard
    import pandas as pd
    from typing import Dict
    from chronostrain.config import cfg
    reference_df = pd.read_csv(ref_index_path, sep="\t")

    json_output_path.parent.mkdir(exist_ok=True, parents=True)
    raw_json_path = json_output_path.with_stem(f'{json_output_path.stem}-1raw')  # first file
    merged_json_path = json_output_path.with_stem(f'{json_output_path.stem}-2overlapmerged')  # second file
    pruned_json_path = json_output_path.with_stem(f'{json_output_path.stem}-3pruned')  # third file

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
                raise FileNotFoundError(f"Reference index pointed to `{seq_path}`, but it does not exist.")
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
        num_threads=num_threads,
        logger=logger
    )

    with open(raw_json_path, 'w') as outfile:
        json.dump(strain_entries, outfile, indent=4)
        logger.info(f"Wrote raw blast DB entries to {raw_json_path}.")

    # ============== Step 2: check for overlaps.
    logger.info("Resolving overlaps.")
    logger.debug(f"Src: {raw_json_path}, Dest: {merged_json_path}")

    from chronostrain.cli.commands.make_db.helpers.resolve_overlaps import find_and_resolve_overlaps
    with open(raw_json_path, "r") as f:
        db_json = json.load(f)
    for strain in db_json:
        find_and_resolve_overlaps(strain, reference_df, logger)
    with open(merged_json_path, 'w') as o:  # dump to JSON.
        json.dump(db_json, o, indent=4)

    if not skip_prune:
        # ============== Step 3: prune using clustering on genomic distances.
        logger.info("Pruning database via clustering")
        logger.debug(f"Src: {merged_json_path}, Dest: {pruned_json_path}")

        # ==== Initialize database instance (to be used for pruning)
        logger.info("Target identity threshold = {}".format(identity_threshold))
        prune_json_db_jaccard(
            src_json_path=merged_json_path,
            tgt_json_path=pruned_json_path,
            cfg=cfg, logger=logger,
            tmp_dir=json_output_path.parent / '__prune_tmp',
            identity_threshold=identity_threshold
        )
        shutil.copy(pruned_json_path, json_output_path)
    else:
        logger.info("Skipping pruning.")
        shutil.copy(merged_json_path, json_output_path)
    logger.info("Done. (Final output = {})".format(json_output_path))


if __name__ == "__main__":
    from chronostrain.logging import create_logger
    my_logger = create_logger("chronostrain.make_db")
    try:
        main(obj=my_logger)
    except Exception as e:
        my_logger.exception(e)
        exit(1)
