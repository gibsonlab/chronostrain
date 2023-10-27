"""
Given that BLAST was run on isolate assembly to identify variants of markers, run multiple alignment (by
appending to existing alignment via MAFFT --add) to obtain hamming distances to existing strains.
"""
from typing import List, Iterator, Tuple, Union, Dict
from dataclasses import dataclass
from pathlib import Path
import csv
import pandas as pd
import click

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.database import StrainDatabase
from chronostrain.database.backend import PandasAssistedBackend
from chronostrain.model import Strain
from chronostrain.util.external import mafft


class IsolateMarker:
    def __init__(self, marker_name: str, nucleotides: Seq):
        self.marker_name = marker_name
        self.nucleotides = nucleotides

    def __repr__(self):
        return f'{self.marker_name}'

    def __str__(self):
        return self.__repr__()


class IsolateAssembly:
    def __init__(self, accession: str, genus: str, species: str, markers: List[IsolateMarker]):
        self.accession = accession
        self.genus = genus
        self.species = species
        self.markers = markers

    def fetch_marker(self, marker_name: str) -> Union[IsolateMarker, None]:
        for marker in self.markers:
            if marker.marker_name == marker_name:
                return marker
        return None

    def __repr__(self):
        return "{}({} {}):[{}]".format(
            self.accession,
            self.genus,
            self.species,
            '|'.join(str(m) for m in self.markers)
        )

    def __str__(self):
        return self.__repr__()


@dataclass
class BlastHit(object):
    line_idx: int
    subj_accession: str
    subj_start: int
    subj_end: int
    subj_len: int
    query_id: str
    query_start: int
    query_end: int
    query_len: int
    strand: str
    evalue: float
    pct_identity: float
    num_gaps: int
    query_coverage_per_hsp: float


# ================== Code for parsing BLAST hits.
def parse_isolate_blast(accession: str, source_dir: Path) -> IsolateAssembly:
    assembly_fasta = source_dir / f'{accession}.fasta'  # file containing the contigs
    genus, species = parse_taxonomy_from_metadata(accession, source_dir)

    blast_tsv = source_dir / f'{accession}.marker_blast.tsv'  # file containing the BLAST hits
    blast_hits = list(parse_blast_hits(blast_tsv))

    markers = []
    for record in SeqIO.parse(assembly_fasta, "fasta"):
        for marker in parse_blast_into_markers(record.id, record.seq, blast_hits):
            markers.append(marker)
    return IsolateAssembly(accession, genus, species, markers)


def parse_blast_into_markers(contig_id: str, contig_seq: Seq, blast_hits: List[BlastHit]) -> Iterator[IsolateMarker]:
    for h in blast_hits:
        if h.query_id != contig_id:
            continue

        nucleotides = contig_seq[h.query_start-1:h.query_end]
        if h.strand == '-':
            nucleotides = nucleotides.reverse_complement()

        yield IsolateMarker(
            marker_name=h.subj_accession,
            nucleotides=nucleotides
        )


def parse_taxonomy_from_metadata(accession: str, source_dir: Path) -> Tuple[str, str]:
    df = pd.read_csv(source_dir / 'metadata.tsv', sep='\t', header=0)
    row = df.loc[df['Accession'] == accession]
    genus = row['Genus'].item()
    species = row['Species'].item()
    return genus, species

def parse_blast_hits(blast_result_path: Path) -> Iterator[BlastHit]:
    with open(blast_result_path, "r") as f:
        blast_result_reader = csv.reader(f, delimiter='\t')
        for row_idx, row in enumerate(blast_result_reader):
            subj_acc, subj_start, subj_end, subj_len, q_seq_id, qstart, qend, qlen, evalue, pident, gaps, qcovhsp = row

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
                
                Swap it so that the coordinates are ALWAYS with respect to the + orientation of the FASTA record.
                """
                subj_start_pos = subj_end
                subj_end_pos = subj_start
                strand = '-'

            yield BlastHit(
                line_idx=row_idx,
                subj_accession=subj_acc,
                subj_start=subj_start_pos,
                subj_end=subj_end_pos,
                subj_len=subj_len,
                query_id=q_seq_id,
                query_start=int(qstart),
                query_end=int(qend),
                query_len=int(qlen),
                strand=strand,
                evalue=float(evalue),
                pct_identity=float(pident),
                num_gaps=int(gaps),
                query_coverage_per_hsp=float(qcovhsp)
            )


# ================================ Code for creating multiple alignments.
def aggregate_distances(isolate: IsolateAssembly,
                        marker_names: List[str],
                        db: StrainDatabase,
                        seed_multiple_alignment_dir: Path,
                        out_dir: Path,
                        n_threads: int = 1) -> pd.DataFrame:
    # Retrieve all markers.
    all_strains = db.all_strains()
    df_entries = []

    from tqdm import tqdm
    pbar = tqdm(marker_names, desc='Alignments')
    for m in pbar:
        pbar.set_postfix({'Marker': m})
        try:
            marker_dists, marker_exists = compute_distances(
                isolate, m,
                db, all_strains,
                seed_multiple_alignment_dir, out_dir,
                n_threads=n_threads
            )
        except NoReferenceAlignmentsException:
            continue

        for s, d in marker_dists.items():
            df_entries.append({'Marker': m, 'ReferenceId': s, 'Distance': d, 'Found': marker_exists})
    return pd.DataFrame(df_entries)


def hamming_dist(x: str, y: str) -> int:
    assert len(x) == len(y)
    return sum(
        1
        for a, b in zip(x, y)
        if a != b
    )


class NoReferenceAlignmentsException(BaseException):
    pass


def compute_distances(
        isolate: IsolateAssembly,
        marker_name: str,
        db: StrainDatabase,
        reference_strains: List[Strain],
        seed_multiple_alignment_dir: Path,
        out_dir: Path,
        n_threads: int = 1
) -> Dict[str, int]:
    marker = isolate.fetch_marker(marker_name)

    # Create a mapping of reference strains to the matching marker ID.
    assert isinstance(db.backend, PandasAssistedBackend)
    marker_to_str = {}
    df = db.backend.strain_df.merge(
        db.backend.marker_df.loc[db.backend.marker_df['MarkerName'] == marker_name],
        on='MarkerIdx',
        how='inner'
    )
    for strain_id, section in df.groupby('StrainId'):
        row = section.head(1)
        marker_id = row['MarkerId'].item()
        marker_to_str[f'{marker_name}|{marker_id}'] = strain_id  # one strain per one marker

    # Run multiple alignment
    original_aln_fasta = seed_multiple_alignment_dir / f'{marker_name}.fasta'
    in_path = out_dir / f'{marker_name}.input.fasta'
    out_path = out_dir / f'{marker_name}.fasta'

    if marker is None:
        # if marker doesn't exist in isolate, compute distances using all-gaps.
        multi_align_len = -1
        for record in SeqIO.parse(original_aln_fasta, 'fasta'):
            multi_align_len = len(record.seq)
        if multi_align_len < 0:
            raise NoReferenceAlignmentsException()

        default_str = '-' * multi_align_len
        ans = {s.id: 0 for s in reference_strains}
        for aln_record in SeqIO.parse(original_aln_fasta, 'fasta'):
            if aln_record.id in marker_to_str:
                strain_id = marker_to_str[aln_record.id]
                ans[strain_id] = hamming_dist(default_str, str(aln_record.seq))
        return ans, False
    else:
        new_record_id = f'{marker.marker_name}_NEW_ISOLATE'
        if not out_path.exists():
            run_mafft(marker, new_record_id, in_path, out_path, original_aln_fasta, n_threads=n_threads)
        isolate_record = find_record_from_fasta(out_path, new_record_id)
        default_dist = hamming_dist(str(isolate_record.seq), '-' * len(isolate_record.seq))
        ans = {s.id: default_dist for s in reference_strains}
        for aln_record in SeqIO.parse(out_path, 'fasta'):
            if aln_record.id in marker_to_str:
                strain_id = marker_to_str[aln_record.id]
                ans[strain_id] = hamming_dist(str(isolate_record.seq), str(aln_record.seq))
        return ans, True


def find_record_from_fasta(f_path: Path, record_id: str) -> SeqRecord:
    for record in SeqIO.parse(f_path, 'fasta'):
        if record.id == record_id:
            return record
    raise ValueError("Record with ID `{}` not found in {}.".format(record_id, f_path))


def run_mafft(marker: IsolateMarker, new_record_id: str, in_path: Path, out_path: Path, original_aln_fasta: Path, n_threads: int = 1):
    with open(in_path, 'w') as f:
        SeqIO.write(
            [SeqRecord(seq=marker.nucleotides, id=new_record_id)],
            f, 'fasta'
        )

    mafft.mafft_add(
        existing_alignment_fasta=original_aln_fasta,
        new_fasta_path=in_path,
        output_path=out_path,
        n_threads=n_threads,
        reorder=False,
        mafft_quiet=True,
        keep_length=False,
        silent=True
    )


def fetch_marker_names(marker_seeds_file: Path) -> List[str]:
    with open(marker_seeds_file, 'rt') as f:
        return [line.split('\t')[0] for line in f]


@click.command()
@click.option(
    '--accession', '-a', 'acc',
    type=str, required=True
)
@click.option(
    '--isolate-dir', '-i', 'isolate_dir',
    type=click.Path(path_type=Path, file_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--marker-seeds', '-m', 'marker_seeds_tsv',
    type=click.Path(path_type=Path, dir_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--seed-multiple-alignments', '-s', 'seed_multiple_alignment_dir',
    type=click.Path(path_type=Path, file_okay=False, exists=True, readable=True), required=True
)
@click.option(
    '--threads', '-t', 'num_threads',
    type=int, required=False, default=1
)
def main(
        acc: str,
        isolate_dir: Path,
        marker_seeds_tsv: Path,
        seed_multiple_alignment_dir: Path,
        num_threads: int
):
    isolate = parse_isolate_blast(
        acc,
        isolate_dir
    )

    marker_names = fetch_marker_names(marker_seeds_tsv)
    out_dir = isolate_dir / f'{acc}_alignments'
    out_dir.mkdir(exist_ok=True, parents=True)

    from chronostrain.config import cfg
    db = cfg.database_cfg.get_database()

    df = aggregate_distances(isolate, marker_names, db, seed_multiple_alignment_dir, out_dir, n_threads=num_threads)
    df.to_csv(out_dir / 'distances.tsv', sep='\t', index=False)


from chronostrain.logging import create_logger
logger = create_logger("chronostrain.compare_isolate_markers")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(e, exc_info=True)
        exit(1)
