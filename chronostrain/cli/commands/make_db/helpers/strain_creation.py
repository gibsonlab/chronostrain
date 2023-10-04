from typing import Dict, List, Any, Optional, Iterator, Union, Callable
from pathlib import Path
import pandas as pd
from logging import Logger
import csv

from intervaltree import IntervalTree

from chronostrain.database.parser.json import SeqEntry
from chronostrain.util.external import blastn, make_blast_db

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
                    min_pct_idty: int,
                    num_ref_genomes: int,
                    num_threads: int,
                    logger: Logger) -> Dict[str, Path]:
    # Run BLAST.
    result_paths: Dict[str, Path] = {}
    result_dir.mkdir(parents=True, exist_ok=True)
    for gene_name, gene_seed_fasta in gene_paths.items():
        max_target_seqs = num_ref_genomes * 10
        logger.debug(f"Running blastn on {gene_name}.")
        blast_result_path = result_dir / f"{gene_name}.tsv"
        # blastn(
        #     db_name=db_name,
        #     db_dir=db_dir,
        #     query_fasta=gene_seed_fasta,
        #     perc_identity_cutoff=min_pct_idty,
        #     out_path=blast_result_path,
        #     num_threads=num_threads,
        #     out_fmt=_BLAST_OUT_FMT,
        #     max_target_seqs=max_target_seqs,
        #     strand="both",
        # )
        result_paths[gene_name] = blast_result_path
    return result_paths


# ===================================================== Parsers and logic

def parse_blast_hits(blast_result_path: Path, min_marker_len: int) -> Iterator[BlastHit]:
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

            yield BlastHit(
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


def create_strain_entries(
        blast_results: Dict[str, Path],
        strain_df: pd.DataFrame,
        min_marker_len: int,
        strain_seqs: Dict[str, List[SeqEntry]],
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
            'seqs': [
                {'accession': seq_entry.accession, 'seq_type': seq_entry.seq_type, 'seq_path': str(seq_entry.seq_path)}
                for seq_entry in strain_seqs[_accession]
            ],
            'markers': []
        }

    strain_entries = {}
    def _tree_overlap_reducer(cur, x):
        cur.append(x)
        return cur

    seqid_to_strain = {
        seq_entry.accession: strain_acc
        for strain_acc, seq_entries in strain_seqs.items()
        for seq_entry in seq_entries
    }

    # ===================== Parse BLAST hits.
    for gene_name, blast_result_path in blast_results.items():
        blast_hits = parse_blast_hits(blast_result_path, min_marker_len)
        gene_intervals = {}

        """
        Parse the entries, recoding the BLAST hit locations along the way. 
        
        As of 2023 Sept. 7, this function now accounts for the possibility of overlapping hits.
        This is a separate handler from what is done in resolve_overlaps.py, which merges overlapping hits of DISTINCT
        genes.
        Instead, this function handles the possibility that a marker seed is a multi-fasta file (with multiplie marker 
        seed sequences that was queried via BLAST.
        If overlapping hits are found for a particular gene (e.g. fimA), then it joins those regions together and calls 
        all of them "fimA". (Note: the strand, by convention, is given by the first observed hit;
        the analysis pipeline uses both + and - strand versions regardless, so this is purely for metadata purposes.)
        """
        logger.debug(f"Parsing BLAST hits for gene `{gene_name}`.")
        # ==== Organize BLAST hits by detecting overlaps via intervaltree.
        for blast_hit in blast_hits:
            fasta_record_id = blast_hit.subj_accession

            start = blast_hit.subj_start
            end = blast_hit.subj_end
            if fasta_record_id not in gene_intervals:
                gene_intervals[fasta_record_id] = IntervalTree()

            """
            interval [start, end+1) --> represents an interval of length end - start + 1 
            (equal to length of covered area reported by BLAST)
            """
            gene_intervals[fasta_record_id][start:end+1] = blast_hit

        # ==== iterate through intervaltree elements to create marker entries.
        for fasta_record_id, tree in gene_intervals.items():
            tree.merge_overlaps(
                data_reducer=_tree_overlap_reducer,
                data_initializer=[],
                strict=False  # adjacent, touching intervals are also merged.
            )

            strain_acc = seqid_to_strain[fasta_record_id]

            if strain_acc not in strain_entries:
                strain_entries[strain_acc] = _entry_initializer(strain_acc)  # could use defaultdict, but it doesn't play so nicely with JSON module sometimes.

            strain_entry = strain_entries[strain_acc]
            for interval in tree:
                if len(interval.data) < 5:
                    gene_id = "{}:BLAST<{}>".format(
                        gene_name,
                        "_".join(str(blast_hit.line_idx) for blast_hit in interval.data)
                    )
                else:
                    gene_id = "{}:BLAST<{}...>".format(gene_name, interval.data[0].line_idx)

                strain_entry['markers'].append(
                    {
                        'id': gene_id,
                        'name': gene_name,
                        'type': 'subseq',
                        'source': fasta_record_id,
                        'start': interval.begin,
                        'end': interval.end - 1,
                        'strand': interval.data[0].strand
                    }
                )

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
        num_threads: int,
        logger: Logger
) -> List[Dict[str, Any]]:
    """
    :return:
    """
    # ============= Create seq catalog.
    strain_seqs = {}
    for _, row in strain_df.iterrows():
        acc = row['Accession']
        seq_path = Path(row['SeqPath'])
        if "*" in seq_path.name:
            # assume each file is a single record. Do a glob search.
            seq_list = [
                SeqEntry(accession=f.stem, seq_type='contig', seq_path=f)
                for f in seq_path.parent.glob(seq_path.name)
            ]
            if len(seq_list) == 0:
                raise ValueError(f"Sequence accession {acc} had an invalid glob {seq_path.name}.")
        else:
            # assume SeqFile is a single-fasta record file, with the same ID as the strain ID.
            seq_list = [SeqEntry(accession=acc, seq_type='chromosome', seq_path=seq_path)]
        strain_seqs[acc] = seq_list

    # ================ Run makeblastdb if necessary.
    blast_db_file = blast_db_dir / f'{blast_db_name}.ndb'
    if not blast_db_file.exists():
        logger.info(f"Blast DB `{blast_db_name}` not found in {blast_db_dir}. Running makeblastdb.")

        # Concatenate FASTA files.
        logger.info("Concatenating input files.")
        concat_fasta = blast_db_dir / f'{blast_db_name}.fasta'
        with open(concat_fasta, "wt") as out_f:
            for acc, seq_entries in strain_seqs.items():
                for seq_entry in seq_entries:
                    assert seq_entry.seq_path is not None
                    with open(seq_entry.seq_path, "rt") as src_f:
                        for line in src_f:
                            out_f.write(line)

        # Invoke makeblastdb.
        make_blast_db(
            input_fasta=concat_fasta,
            db_dir=blast_db_dir,
            db_name=blast_db_name,
            is_nucleotide=True,
            title=blast_db_name,
            parse_seqids=True
        )

        # Clean up.
        logger.info("Cleaning up.")
        concat_fasta.unlink()  # clean up.

    # ================= run BLAST.
    blast_results = run_blast_local(
        db_dir=blast_db_dir,
        db_name=blast_db_name,
        result_dir=blast_result_dir,
        gene_paths=gene_paths,
        min_pct_idty=min_pct_idty,
        num_ref_genomes=strain_df.shape[0],
        num_threads=num_threads,
        logger=logger,
    )

    return create_strain_entries(blast_results, strain_df, min_marker_len, strain_seqs, logger)
