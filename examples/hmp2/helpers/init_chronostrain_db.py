"""
Script which creates a chronostrain db of all UNIREF-specified marker genes.
Works in 3 steps:
    1) For each marker gene name, extract its annotation from reference K-12 genome.
    2) Run BLAST to find all matching hits (even partial hits) of marker gene sequence to each strain assembly
        (with blastn configured to allow, on average, up to 10 hits per genome).
    3) Convert BLAST results into chronostrain JSON database specification.
"""
import argparse
import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import pandas as pd

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from chronostrain.util.entrez import fetch_fasta
from chronostrain.util.external import make_blast_db, blastn

from typing import List, Dict, Any, Tuple, Iterator, Callable
from intervaltree import IntervalTree

from chronostrain.config import cfg
from chronostrain.util.io import read_seq_file
from chronostrain.logging import create_logger
logger = create_logger("chronostrain.init_db")


READ_LEN = 150


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create chronostrain database file from specified gene_info_uniref marker CSV."
    )

    # Input specification.
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='<Required> The path to the target output chronostrain db json file.')
    parser.add_argument('-r', '--refseq_dir', required=True, type=str,
                        help='<Required> The strainGE database directory.')

    return parser.parse_args()


# ============================= Common resources
def strain_seq_dir(strain_id: str) -> Path:
    return cfg.database_cfg.data_dir / "assemblies" / strain_id


# ============================= Creation of database configuration (Indexing + partial JSON creation)
def parse_assembly_report_filepath(assembly_report_path: Path) -> Tuple[str, Path]:
    suffix = '_assembly_report.txt'
    basename = assembly_report_path.name[:-len(suffix)]
    tokens = basename.split('_')

    if tokens[0] != "GCF":
        raise RuntimeError(f"Unexpected naming format for file `{assembly_report_path}`.")

    gcf_id = f"{tokens[0]}_{tokens[1]}"
    fasta_path = assembly_report_path.parent / f"{basename}_genomic.fna.gz"

    if not fasta_path.exists():
        raise FileNotFoundError(
            f"Expected {fasta_path.name} relative to {assembly_report_path.name}, but does not exist."
        )

    return gcf_id, fasta_path


def perform_indexing(refseq_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all existing assembly files in the directory hierarchy (downloaded from NCBI).
    :return: a pair of dataframes:
    1) A description of each strain's label,
    2) A complete index of all relevant chromosomes/scaffolds/contigs etc.
    """
    strain_entries = []
    seq_entries = []

    if not refseq_dir.is_dir():
        raise RuntimeError(f"Provided path `{refseq_dir}` is not a directory.")
    for genus_dir in (refseq_dir / "human_readable" / "refseq" / "bacteria").iterdir():
        if not genus_dir.is_dir():
            continue

        genus = genus_dir.name
        for species_dir in genus_dir.iterdir():
            if not species_dir.is_dir():
                continue

            species = species_dir.name
            if species == 'sp.':
                species = 'sp'

            logger.info(f"Searching through {genus} {species}...")

            for strain_dir in species_dir.iterdir():
                if not strain_dir.is_dir():
                    continue

                strain_name = strain_dir.name
                target_files = list(strain_dir.glob("*_assembly_report.txt"))

                for fpath in target_files:
                    gcf_id, _ = parse_assembly_report_filepath(fpath)

                    logger.info(f"Parsing strain `{strain_name}`, (target file: {fpath})")
                    strain_entries.append({
                        'StrainId': gcf_id,
                        'Genus': genus,
                        'Species': species,
                        'Name': strain_name
                    })

                    for seq_entry in parse_assembly_report(Path(fpath)):
                        seq_entries.append(seq_entry)
    return pd.DataFrame(strain_entries), pd.DataFrame(seq_entries)


class AssemblyReportEntry:
    """
    Represents a single line in <gcf_id>_assembly_report.txt.
    """
    def __init__(self, line):
        tokens = line.split('\t')
        self.seq_name = tokens[0]
        self.seq_role = tokens[1]
        self.asgn_mol = tokens[2]
        self.asgn_mol_loctype = tokens[3]
        self.gb_accession = tokens[4]
        self.relationship = tokens[5]
        self.refseq_accession = tokens[6]
        self.assembly_unit = tokens[7]
        self.seq_len = tokens[8]
        self.ucsc_style_name = tokens[9]

    @property
    def is_chromosome(self) -> bool:
        return (self.seq_role == 'assembled-molecule') and (self.asgn_mol_loctype == 'Chromosome')

    @property
    def is_scaffold(self) -> bool:
        return 'scaffold' in self.seq_role

    @property
    def is_contig(self) -> bool:
        return 'contig' in self.seq_role


def parse_assembly_report(report_path: Path) -> Iterator[Dict]:
    """
    Parse assembly report, save all relevant seqs to separate files and them by returning a DataFrame dictionary entry.
    """
    gcf_id, refseq_fasta_path = parse_assembly_report_filepath(report_path)

    seq_records: Dict[str, SeqRecord] = {}
    for record in read_seq_file(refseq_fasta_path, file_format='fasta'):
        refseq_accession = record.id.split(' ')[0]
        seq_records[refseq_accession] = record

    with open(report_path, 'r') as report:
        n_chromosomes = 0
        n_scaffolds = 0
        n_contigs = 0

        for line in report:
            if line.startswith('#'):
                continue

            entry = AssemblyReportEntry(line)
            if entry.is_chromosome:
                logger.debug(f"Found chromosome `{entry.refseq_accession}`")
                if entry.refseq_accession not in seq_records:
                    raise KeyError(
                        f"Could not find matching chromosome `{entry.refseq_accession}` in {refseq_fasta_path}."
                    )

                seq_path = report_path.parent / f"{entry.refseq_accession}.fasta"
                SeqIO.write([seq_records[entry.refseq_accession]], seq_path, 'fasta')
                yield {
                    "StrainId": gcf_id,
                    "SeqAccession": entry.refseq_accession,
                    "SeqPath": seq_path,
                    "SeqType": "chromosome"
                }
                n_chromosomes += 1

            if entry.is_scaffold:
                """
                Each scaffold piece goes into its own FASTA file.
                """
                logger.debug(f"Found scaffold `{entry.refseq_accession}`")
                if entry.refseq_accession not in seq_records:
                    raise KeyError(f"Could not find matching scaffold `{entry.refseq_accession}` in {seq_path}.")

                seq_path = report_path.parent / f"{entry.refseq_accession}.fasta"
                SeqIO.write([seq_records[entry.refseq_accession]], seq_path, 'fasta')
                yield {
                    "StrainId": gcf_id,
                    "SeqAccession": entry.refseq_accession,
                    "SeqPath": seq_path,
                    "SeqType": "scaffold"
                }
                n_scaffolds += 1

            if entry.is_contig:
                """
                Each contig piece goes into its own FASTA file.
                """
                logger.debug(f"Found contig `{entry.refseq_accession}`")
                if entry.refseq_accession not in seq_records:
                    raise KeyError(f"Could not find matching contig `{entry.refseq_accession}` in {seq_path}.")

                seq_path = report_path.parent / f"{entry.refseq_accession}.fasta"
                SeqIO.write([seq_records[entry.refseq_accession]], seq_path, 'fasta')
                yield {
                    "StrainId": gcf_id,
                    "SeqAccession": entry.refseq_accession,
                    "SeqPath": seq_path,
                    "SeqType": "contig"
                }
                n_contigs += 1

        logger.info(f"# chromosomes: {n_chromosomes}, # scaffolds: {n_scaffolds}, # contigs: {n_contigs}")


# ============= Browse reference genes (assuming they have already been extracted.)
def reference_marker_genes(genus: str, species: str) -> Iterator[Tuple[str, Path]]:
    dir_path = cfg.database_cfg.data_dir / "reference" / genus / species
    for fasta_path in glob.glob(str(dir_path / "*.fasta")):
        fasta_path = Path(fasta_path)
        gene_name = fasta_path.stem
        yield gene_name, fasta_path


# ============= Rest of initialization (Run BLAST)
def extract_ungapped_subseq(strain_id: str, seq_accession: str, blast_hit: 'BlastHit') -> Tuple[int, int]:
    """
    Strategy for gapped scaffolds: If the BLAST hit contains gaps, take the largest ungapped sub-region.
    :return: The start and end position (note: 1-indexed, inclusive) of the longest consecutive ungapped substring.
    """
    subseq = str(
        next(read_seq_file(
            fetch_fasta(seq_accession, strain_seq_dir(strain_id)),
            "fasta"
        )).seq[blast_hit.subj_start - 1:blast_hit.subj_end]
    )

    # Extract the largest.
    best_token_len = -1
    best_left_idx = -1
    running_sum = 0
    for tok_idx, token in enumerate(subseq.split('N')):
        if len(token) > best_token_len:
            best_token_len = len(token)
            best_left_idx = tok_idx + running_sum
        running_sum += len(token)

    if best_token_len <= 0:
        raise RuntimeError("Subseq for BLAST hit contained only gaps.")

    overall_start = blast_hit.subj_start + best_left_idx
    return overall_start, overall_start - 1 + best_token_len


def create_chronostrain_db(
        strain_index: pd.DataFrame,
        seq_index: pd.DataFrame,
        output_path: Path
) -> List[Dict[str, Any]]:
    """
    :param strain_index: A DataFrame with 4 columns: StrainId, Genus, Species, Name.
    :param seq_index: A DataFrame with 4 columns: StrainId, SeqAccession, SeqPath, SeqType.
    :param output_path:
    :return:
    """

    # ======== BLAST configuration
    blast_db_dir = output_path.parent / "blast"
    blast_db_name = "db_bacteria"
    blast_db_title = "\"Bacteria (metaphlan markers, NCBI complete)\""
    blast_fasta_path = blast_db_dir / "genomes.fasta"
    blast_result_dir = output_path.parent / "blast_results"
    logger.info("BLAST\n\tdatabase location: {}\n\tresults directory: {}".format(
        str(blast_db_dir),
        str(blast_result_dir)
    ))

    # ========= Initialize BLAST database.
    blast_db_dir.mkdir(parents=True, exist_ok=True)
    strain_fasta_files = []
    json_strain_entries = []
    for idx, row in strain_index.iterrows():
        strain_id = row['StrainId']
        json_seq_entries, seq_paths = extract_strain_seqs(seq_index, strain_id)
        if len(json_seq_entries) == 0:
            logger.info(f"No usable sequence assemblies found for `{strain_id}`.")
            continue

        json_strain_entries.append({
            'id': strain_id,
            'genus': row['Genus'],
            'species': row['Species'],
            'name': row['Name'],
            'seqs': json_seq_entries,
            'markers': []
        })
        strain_fasta_files += seq_paths

    logger.info('Concatenating {} files, in preparation for makeblastdb.'.format(len(strain_fasta_files)))
    with open(blast_fasta_path, 'w') as genome_fasta_file:
        for fpath in strain_fasta_files:
            with open(fpath, 'r') as in_file:
                for line in in_file:
                    genome_fasta_file.write(line)

    make_blast_db(
        blast_fasta_path, blast_db_dir, blast_db_name,
        is_nucleotide=True, title=blast_db_title, parse_seqids=True
    )

    Path(blast_fasta_path).unlink()  # clean up large fasta file.

    def all_reference_genes() -> Iterator[Tuple[str, Path]]:
        for genus in strain_index['Genus'].unique():
            for species in strain_index.loc[strain_index['Genus'] == genus, 'Species'].unique():
                yield from reference_marker_genes(genus, species)

    # ========= Run BLAST to find marker genes.
    blast_hits: Dict[str, Dict[str, List['BlastHit']]] = {}
    gene_names: List[str] = []
    blast_result_dir.mkdir(parents=True, exist_ok=True)
    for gene_name, ref_gene_path in all_reference_genes():
        blast_hits[gene_name] = defaultdict()
        logger.info(f"Running blastn on {gene_name}.")
        blast_result_path = blast_result_dir / f"{gene_name}.tsv"
        blastn(
            db_name=blast_db_name,
            db_dir=blast_db_dir,
            query_fasta=ref_gene_path,
            evalue_max=1e-3,
            out_path=blast_result_path,
            num_threads=cfg.model_cfg.num_cores,
            out_fmt="6 saccver sstart send qstart qend evalue bitscore pident gaps qcovhsp",
            max_target_seqs=10 * seq_index.shape[0],  # A generous value, 10 hits per genome
            strand="both"
        )

        gene_names.append(gene_name)
        blast_hits[gene_name] = parse_blast_hits(blast_result_path)

    for json_strain_entry in json_strain_entries:
        logger.debug(f"Looking for BLAST hits for strain `{json_strain_entry['id']}`.")
        marker_objs = blast_hits_into_markers(
            strain_id=json_strain_entry['id'],
            seq_accessions=[seq['accession'] for seq in json_strain_entry['seqs']],
            genes=gene_names,
            get_blast_hits=lambda acc, gene: blast_hits[gene][acc]
        )
        json_strain_entry['markers'] = marker_objs
    return prune_entries(json_strain_entries)


def blast_hits_into_markers(strain_id: str,
                            seq_accessions: List[str],
                            genes: List[str],
                            get_blast_hits: Callable[[str, str], List['BlastHit']]) -> List[Dict]:
    """
    For the provided strain (implicitly defined by the collection `seq_accessions`), collect all of the BLAST hits
    assorted by genes.

    Performs an additional check to see whether BLAST hits are overlapping. In that scenario, merges them into one
    large gene.

    :return: A list of JSON objects (dictionaries) representing marker entries.
    """
    def add_to_tree(t: IntervalTree, start_loc: int, end_loc: int, strand: str, gene_name: str, blast_idx: int):
        interval_len = end_loc - start_loc + 1
        if interval_len >= READ_LEN:
            t[start_loc:end_loc+1] = (start_loc, end_loc, strand, gene_name, blast_idx)
        else:
            logger.warning(f"Requested interval is too short (len = {interval_len}). Skipping.")

    marker_objs = []
    for seq_accession in seq_accessions:
        tree = IntervalTree()
        for gene_name in genes:
            for blast_hit in get_blast_hits(seq_accession, gene_name):
                start, end = extract_ungapped_subseq(strain_id, seq_accession, blast_hit)
                hit_len = end - start + 1

                overlapping_hits = tree[start:end+1]
                if len(overlapping_hits) > 0:
                    # Special scenario: remove overlapping regions of hits.
                    logger.info("Found {} hits that overlap with blast hit {}({}--{}).".format(
                        len(overlapping_hits),
                        gene_name,
                        start, end
                    ))

                for other_hit in overlapping_hits:
                    _start_other, _end_other, _strand_other, _name_other, _blast_idx_other = other_hit.data
                    logger.info("Target hit: {}({}--{})".format(_name_other, _start_other, _end_other))

                    _len_other = _end_other - _start_other + 1
                    if _len_other > hit_len:
                        # Chop off overlapping region from this hit.
                        if _start_other <= start <= end <= _end_other:
                            end = start - 1
                        elif _start_other <= start <= _end_other:
                            start = _end_other + 1
                        elif _start_other <= end <= _end_other:
                            end = _start_other - 1
                        else:
                            raise RuntimeError("Unknown overlap scenario.")
                        hit_len = end - start + 1
                    else:
                        # Chop off overlapping region from other hit.
                        if start <= _start_other <= _end_other <= end:
                            _end_other = _start_other - 1
                        elif start <= _start_other <= end:
                            _start_other = end + 1
                        elif start <= _end_other <= end:
                            _end_other = start - 1
                        else:
                            raise RuntimeError("Unknown overlap scenario.")
                        tree.remove(other_hit)
                        add_to_tree(tree, _start_other, _end_other, _strand_other, _name_other, _blast_idx_other)

                # Default scenario
                add_to_tree(tree, start, end, blast_hit.strand, gene_name, blast_hit.line_idx)

        # Now, hits are guaranteed to be non-overlapping. Instantiate the objects.
        for node in tree:
            start_loc, end_loc, strand, gene_name, line_idx = node.data
            gene_id = f"{gene_name}#{line_idx}"
            marker_objs.append({
                'id': gene_id,
                'name': gene_name,
                'type': 'subseq',
                'source': seq_accession,
                'start': start_loc,
                'end': end_loc,
                'strand': strand,
                'canonical': False
            })

        if len(tree) > 0:
            logger.debug(f"Parsed {len(tree)} hits for sequence accession {seq_accession}.")
    return marker_objs


def extract_strain_seqs(seq_df: pd.DataFrame, strain_id: str) -> Tuple[List[Dict], List[Path]]:
    """
    Extract all sequences from seq_df corresponding to strain_id.
    Creates a symbolic link to each relevant seq file into chronostrain's configured database directory, and
    adds the corresponding index entry to the JSON object.
    """
    section = seq_df.loc[seq_df['StrainId'] == strain_id, ['SeqAccession', 'SeqPath', 'SeqType']]
    json_strain_seqs = []
    seq_paths = []
    for _, row in section.iterrows():
        seq_accession = row['SeqAccession']
        seq_path = Path(row['SeqPath'])
        seq_type = row['SeqType']

        # Add the JSON attribute.
        json_strain_seqs.append({
            'accession': seq_accession,
            'seq_type': seq_type
        })

        # Create symbolic link to sequence file in database directory.
        symlink_path = strain_seq_dir(strain_id) / f"{seq_accession}.fasta"
        if symlink_path.exists():
            logger.debug(f"Path {symlink_path} already exists.")
        else:
            symlink_path.parent.mkdir(exist_ok=True, parents=True)
            symlink_path.symlink_to(seq_path)
            logger.debug(f"Created symlink {symlink_path} -> {seq_path}")
        seq_paths.append(symlink_path)
    return json_strain_seqs, seq_paths


def prune_entries(strain_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Delete all strain entries with zero markers.
    """
    for strain_entry in strain_entries:
        if len(strain_entry['markers']) == 0:
            logger.info("No markers found for "
                        f"{strain_entry['genus']} {strain_entry['species']}, "
                        f"{strain_entry['name']}.")
    return [
        strain_entry
        for strain_entry in strain_entries
        if len(strain_entry['markers']) > 0
    ]


@dataclass
class BlastHit(object):
    line_idx: int
    subj_accession: str
    subj_start: int
    subj_end: int
    query_start: int
    query_end: int
    strand: str
    evalue: float
    bitscore: float
    pct_identity: float
    num_gaps: int
    query_coverage_per_hsp: float

    def __repr__(self):
        return "BLAST#{}<{}:{}>".format(
            self.line_idx,
            self.subj_start,
            self.subj_end
        )

    def __str__(self):
        return self.__repr__()


def parse_blast_hits(blast_result_path: Path) -> Dict[str, List[BlastHit]]:
    """
    :return: A dictionary of blast hits categorized by database sequence, e.g.
     <sequence accession> -> <Blast hits on that accession>
    """
    blast_hits_by_acc = defaultdict(list)
    with open(blast_result_path, "r") as f:
        blast_result_reader = csv.reader(f, delimiter='\t')
        for row_idx, row in enumerate(blast_result_reader):

            subj_acc, subj_start, subj_end, qstart, qend, evalue, bitscore, pident, gaps, qcovhsp = row

            subj_start = int(subj_start)
            subj_end = int(subj_end)

            if subj_start < subj_end:
                start_pos = subj_start
                end_pos = subj_end
                strand = '+'
            else:
                start_pos = subj_end
                end_pos = subj_start
                strand = '-'

            if end_pos - start_pos + 1 < READ_LEN:
                continue

            blast_hits_by_acc[subj_acc].append(
                BlastHit(
                    row_idx,
                    subj_acc,
                    start_pos,
                    end_pos,
                    int(qstart),
                    int(qend),
                    strand,
                    float(evalue),
                    float(bitscore),
                    float(pident),
                    int(gaps),
                    float(qcovhsp)
                )
            )
    return blast_hits_by_acc

def main():
    args = parse_args()
    output_path = Path(args.output_path)
    seq_dir = Path(args.refseq_dir)

    # ================= Indexing of refseqs
    logger.info(f"Indexing refseqs located in {seq_dir}")
    strain_df, seq_df = perform_indexing(seq_dir)

    # Save both dataframes to disk.
    strain_index_path = seq_dir / "strains.tsv"
    strain_df.to_csv(strain_index_path, sep='\t', index=False)
    logger.info(f"Wrote strain index to {str(strain_index_path)}.")
    seq_index_path = seq_dir / "seqs.tsv"
    seq_df.to_csv(seq_index_path, sep='\t', index=False)
    logger.info(f"Wrote seq index to {str(seq_index_path)}.")

    # ================= Compile into JSON.
    logger.info("Creating JSON entries.")
    object_entries = create_chronostrain_db(strain_df, seq_df, output_path)

    with open(output_path, 'w') as outfile:
        json.dump(object_entries, outfile, indent=4)
    logger.info(f"Wrote output to {str(output_path)}.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        logger.exception(e)
        raise
