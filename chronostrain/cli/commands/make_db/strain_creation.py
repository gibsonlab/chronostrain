from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
from logging import Logger
from collections import defaultdict
import csv

from chronostrain.config import cfg
from chronostrain.util.io import read_seq_file
from chronostrain.util.external import blastn


# ===================================================== BLAST-specific definitions

_BLAST_OUT_FMT = "6 saccver sstart send slen qstart qend evalue pident gaps qcovhsp"


class BlastHit(object):
    def __init__(self,
                 line_idx: int,
                 subj_accession: str,
                 subj_start: int,
                 subj_end: int,
                 subj_len: int,
                 query_start: int,
                 query_end: int,
                 strand: str,
                 evalue: float,
                 pct_identity: float,
                 num_gaps: int,
                 query_coverage_per_hsp: float,
                 ):
        self.line_idx = line_idx
        self.subj_accession = subj_accession
        self.subj_start = subj_start
        self.subj_end = subj_end
        self.subj_len = subj_len
        self.query_start = query_start
        self.query_end = query_end
        self.strand = strand
        self.evalue = evalue
        self.pct_identity = pct_identity
        self.num_gaps = num_gaps
        self.query_coverage_per_hsp = query_coverage_per_hsp


def run_blast_local(db_dir: Optional[Path],
                    db_name: str,
                    result_dir: Path,
                    gene_paths: Dict[str, Path],
                    max_target_seqs: int,
                    min_pct_idty: int,
                    logger: Logger) -> Dict[str, Path]:
    # Run BLAST.
    result_paths: Dict[str, Path] = {}
    result_dir.mkdir(parents=True, exist_ok=True)
    for gene_name, ref_gene_path in gene_paths.items():
        logger.info(f"Running blastn on {gene_name}.")
        blast_result_path = result_dir / f"{gene_name}.tsv"
        blastn(
            db_name=db_name,
            db_dir=db_dir,
            query_fasta=ref_gene_path,
            perc_identity_cutoff=min_pct_idty,
            out_path=blast_result_path,
            num_threads=cfg.model_cfg.num_cores,
            out_fmt=_BLAST_OUT_FMT,
            max_target_seqs=max_target_seqs,
            strand="both"
        )
        result_paths[gene_name] = blast_result_path
    return result_paths


# ===================================================== Parsers and logic

def parse_blast_hits(blast_result_path: Path, min_marker_len: int) -> Dict[str, List[BlastHit]]:
    accession_to_positions: Dict[str, List[BlastHit]] = defaultdict(list)
    with open(blast_result_path, "r") as f:
        blast_result_reader = csv.reader(f, delimiter='\t')
        for row_idx, row in enumerate(blast_result_reader):
            subj_acc, subj_start, subj_end, subj_len, qstart, qend, evalue, pident, gaps, qcovhsp = row

            subj_start = int(subj_start)
            subj_end = int(subj_end)
            subj_len = int(subj_len)

            if subj_start < subj_end:
                subj_start_pos = subj_start
                subj_end_pos = subj_end
                strand = '+'
            else:
                """
                BLAST convention: if hit is on the minus strand, start > end (to indicate the negative direction).
                *** IMPORTANT *** --- note that the actual VALUES of (start, end) is with respect to the plus strand.
                """
                subj_start_pos = subj_end
                subj_end_pos = subj_start
                strand = '-'

            if subj_end_pos - subj_start_pos + 1 < min_marker_len:
                continue

            accession_to_positions[subj_acc].append(
                BlastHit(
                    line_idx=row_idx,
                    subj_accession=subj_acc,
                    subj_start=subj_start_pos,
                    subj_end=subj_end_pos,
                    subj_len=subj_len,
                    query_start=int(qstart),
                    query_end=int(qend),
                    strand=strand,
                    evalue=float(evalue),
                    pct_identity=float(pident),
                    num_gaps=int(gaps),
                    query_coverage_per_hsp=float(qcovhsp)
                )
            )
    return accession_to_positions


def create_strain_entries(
        blast_results: Dict[str, Path],
        ref_gene_paths: Dict[str, Path],
        strain_df: pd.DataFrame,
        min_marker_len: int,
        logger: Logger
):
    def _entry_initializer(_accession):
        strain_row = strain_df.loc[strain_df['Accession'] == _accession, :].head(1)
        subj_genus = strain_row['Genus'].item()
        subj_species = strain_row['Species'].item()
        subj_strain_name = strain_row['Strain'].item()
        genome_length = strain_row['ChromosomeLen'].item()

        return {
            'id': _accession,
            'genus': subj_genus,
            'species': subj_species,
            'name': subj_strain_name,
            'genome_length': genome_length,
            'seqs': [{'accession': _accession, 'seq_type': 'chromosome'}],
            'markers': []
        }

    strain_entries = {}

    # ===================== Parse BLAST hits.
    for gene_name, blast_result_path in blast_results.items():
        blast_hits = parse_blast_hits(blast_result_path, min_marker_len)

        # Parse the entries.
        canonical_gene_found = False
        ref_gene_path = ref_gene_paths[gene_name]
        ref_gene_len = len(next(iter(read_seq_file(ref_gene_path, 'fasta'))).seq)
        min_canonical_length = ref_gene_len * 0.95

        logger.debug(f"Parsing BLAST hits for gene `{gene_name}`.")
        for subj_acc in blast_hits.keys():
            if subj_acc not in strain_entries:
                strain_entries[subj_acc] = _entry_initializer(subj_acc)

            strain_entry = strain_entries[subj_acc]
            seq_accession = strain_entry['seqs'][0]['accession']
            for blast_hit in blast_hits[seq_accession]:
                is_canonical = (
                        (blast_hit.subj_end - blast_hit.subj_start) >= min_canonical_length
                        and
                        not canonical_gene_found
                )

                gene_id = f"{gene_name}_BLASTIDX_{blast_hit.line_idx}"
                strain_entry['markers'].append(
                    {
                        'id': gene_id,
                        'name': gene_name,
                        'type': 'subseq',
                        'source': seq_accession,
                        'start': blast_hit.subj_start,
                        'end': blast_hit.subj_end,
                        'strand': blast_hit.strand,
                        'canonical': is_canonical
                    }
                )
                canonical_gene_found = canonical_gene_found or is_canonical

    return prune_entries([entry for _, entry in strain_entries.items()], logger)


def prune_entries(strain_entries: List[Dict[str, Any]], logger: Logger) -> List[Dict[str, Any]]:
    """
    Given a list of strain definitions, delete entries that don't have any marker sequences.
    :param strain_entries: List of Strain entries (as defined in the JSON specification).
    :return: A pruned list of entries
    """
    for strain_entry in strain_entries:
        if len(strain_entry['markers']) == 0:
            logger.info("No markers found for "
                        f"{strain_entry['genus']} {strain_entry['species']} "
                        f"{strain_entry['name']} "
                        f"(ID {strain_entry['id']}).")
    return [
        strain_entry
        for strain_entry in strain_entries
        if len(strain_entry['markers']) > 0
    ]


def create_chronostrain_db(
        blast_result_dir: Path,
        strain_df: pd.DataFrame,
        gene_paths: Dict[str, Path],
        blast_db_dir: Path,
        blast_db_name: str,
        min_pct_idty: int,
        min_marker_len: int,
        logger: Logger
) -> List[Dict[str, Any]]:
    """
    :return:
    """
    num_ref_genomes = strain_df.shape[0]

    blast_results = run_blast_local(
        db_dir=blast_db_dir,
        db_name=blast_db_name,
        result_dir=blast_result_dir,
        gene_paths=gene_paths,
        min_pct_idty=min_pct_idty,
        max_target_seqs=10 * num_ref_genomes,
        logger=logger
    )

    return create_strain_entries(blast_results, gene_paths, strain_df, min_marker_len, logger)
