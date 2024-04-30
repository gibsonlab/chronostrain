from pathlib import Path
from typing import List, Dict, Set, Tuple
from logging import Logger

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.database import StrainDatabase
from chronostrain.model import Marker
from chronostrain.util.external import mafft_global

from chronostrain.config import cfg


def marker_concatenated_multiple_alignments(db: StrainDatabase, out_path: Path, marker_names: List[str], logger: Logger):
    """
    Generates a single FASTA file containing the concatenation of the multiple alignments of each marker gene.
    If a gene is missing from a strain, gaps are appended instead.
    If multiple hits are found, then the first available one is used (found in the same order as BLAST hits).
    """

    """
    1. all_marker_alignments: contains the mapping (gene_name) -> (marker_id) -> (Seq) that contains the 
    multiple alignment.
    2. marker_assignments: contains the mapping of (strain) -> (gene_name) -> (marker) that was included in the 
    alignment.
    """
    all_marker_alignments, marker_assignments = get_all_alignments(db, out_path.parent / out_path.stem, marker_names, logger)
    marker_names = [m for m in marker_names if len(all_marker_alignments[m]) > 0]

    records: List[SeqRecord] = []
    for strain in db.all_strains():
        seqs_to_concat = []

        # Concatenate the alignment sequence in the particular order.
        target_gene_ids = []
        for gene_name in marker_names:
            record_map = all_marker_alignments[gene_name]
            if gene_name not in marker_assignments[strain.id]:
                _, example_record = next(iter(record_map.items()))
                aln_len = len(example_record.seq)
                seqs_to_concat.append(
                    "".join('-' for _ in range(aln_len))
                )
                target_gene_ids.append("-")
            else:
                target_marker = marker_assignments[strain.id][gene_name]
                record = record_map[target_marker.id]
                seqs_to_concat.append(
                    str(record.seq)
                )
                target_gene_ids.append(target_marker.id)
        records.append(
            SeqRecord(
                Seq("".join(seqs_to_concat)),
                id=strain.id,
                description=f"{strain.name}:" + "|".join(target_gene_ids)
            )
        )

    SeqIO.write(
        records, out_path, "fasta"
    )


def multi_align_markers(output_path: Path, markers: List[Marker], n_threads: int = 1) -> Dict[str, SeqRecord]:
    input_fasta_path = output_path.with_suffix('.input.fasta')

    SeqIO.write(
        [marker.to_seqrecord() for marker in markers],
        input_fasta_path,
        "fasta"
    )

    mafft_global(
        input_fasta_path=input_fasta_path,
        output_path=output_path,
        n_threads=n_threads,
        auto=True,
        max_iterates=1000
    )

    ids_to_records = {}
    for record in SeqIO.parse(output_path, format='fasta'):
        try:
            marker_name, marker_id = Marker.parse_seqrecord_id(record.id)
        except ValueError:
            raise ValueError("Couldn't parse record ID {}".format(record.id)) from None
        ids_to_records[marker_id] = record
    return ids_to_records


def get_all_alignments(
        db: StrainDatabase, work_dir: Path, marker_names: List[str], logger: Logger
) -> Tuple[
    Dict[str, Dict[str, SeqRecord]],
    Dict[str, Dict[str, Marker]]
]:
    work_dir.mkdir(exist_ok=True, parents=True)
    all_alignments = {}
    def _filter_markers(s, g):
        best_len = 0
        best_marker = None
        for marker in s.markers:
            if marker.name == g:  # pick the longest one in the list.
                if best_len < len(marker):
                    best_len = len(marker)
                    best_marker = marker
        if best_marker is not None:
            return best_marker
        raise ValueError("Not found!")

    gene_assignments = {
        s.id: {}
        for s in db.all_strains()
    }

    for gene_name in marker_names:
        logger.info(f"Aligning instances of {gene_name}")
        markers = []
        for s in db.all_strains():
            try:
                m = _filter_markers(s, gene_name)
                markers.append(m)
                gene_assignments[s.id][gene_name] = m
            except ValueError:
                pass

        alignment_records = multi_align_markers(
            output_path=work_dir / f"{gene_name}.fasta",
            markers=markers,
            n_threads=cfg.model_cfg.num_cores
        )

        all_alignments[gene_name] = alignment_records
    return all_alignments, gene_assignments
