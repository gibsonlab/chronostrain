import argparse
from pathlib import Path
from typing import List, Dict

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain import cfg
from chronostrain.database import StrainDatabase
from chronostrain.model import Marker
from chronostrain.util.external import mafft_global


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple alignments of all marker genes found in database."
    )

    # Input specification.
    parser.add_argument('-o', '--output_path', required=True, type=str,
                        help='<Required> The target output path to write the resulting tree to (in newick format).')
    return parser.parse_args()


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
        strain_id, marker_name, marker_id = Marker.parse_seqrecord_id(record.id)
        ids_to_records[marker_id] = record
    return ids_to_records


def get_all_alignments(db: StrainDatabase, work_dir: Path) -> Dict[str, Dict[str, SeqRecord]]:
    all_alignments = {}
    for gene_name in db.all_marker_names():
        alignment_records = multi_align_markers(
            output_path=work_dir / f"{gene_name}.fasta",
            markers=db.get_markers_by_name(gene_name),
            n_threads=cfg.model_cfg.num_cores
        )

        all_alignments[gene_name] = alignment_records
    return all_alignments


def get_concatenated_alignments(db: StrainDatabase, out_path: Path):
    """
    Generates a single FASTA file containing the concatenation of the multiple alignments of each marker gene.
    If a gene is missing from a strain, gaps are appended instead.
    If multiple hits are found, then the first available one is used (found in the same order as BLAST hits).
    """
    all_marker_alignments = get_all_alignments(db, out_path.parent)

    records: List[SeqRecord] = []
    for strain in db.all_strains():
        seqs_to_concat = []
        gene_names = list(all_marker_alignments.keys())

        strain_marker_map: Dict[str, Marker] = {}
        for marker in strain.markers:
            # Append the first available marker with matching gene name.
            if marker.name not in strain_marker_map:
                strain_marker_map[marker.name] = marker

        gene_ids = []
        for gene_name in gene_names:
            record_map = all_marker_alignments[gene_name]

            if gene_name not in strain_marker_map:
                _, example_record = next(iter(record_map.items()))
                aln_len = len(example_record.seq)
                seqs_to_concat.append(
                    "".join('-' for _ in range(aln_len))
                )
                gene_ids.append("-")
            else:
                target_marker = strain_marker_map[gene_name]
                record = record_map[target_marker.id]
                seqs_to_concat.append(
                    str(record.seq)
                )
                gene_ids.append(target_marker.id)
        records.append(
            SeqRecord(
                Seq("".join(seqs_to_concat)),
                id=strain.id,
                description="|".join(gene_ids)
            )
        )

        SeqIO.write(
            records, out_path, "fasta"
        )


def main():
    args = parse_args()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    db = cfg.database_cfg.get_database()
    get_concatenated_alignments(db, out_path)


if __name__ == "__main__":
    main()
