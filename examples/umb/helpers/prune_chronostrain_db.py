import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
import json

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import numpy as np
from chronostrain import cfg
from chronostrain.util.sequences import nucleotides_to_z4
from chronostrain.database import JSONStrainDatabase, StrainDatabase
from chronostrain.model import Marker, Strain
from chronostrain.util.external import mafft_global


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster strains by similarity (in hamming distance of concatenated marker alignments)"
    )

    # Input specification.
    parser.add_argument('--input_json', required=True, type=str,
                        help='<Required> The input database JSON file.')
    parser.add_argument('--output_json', required=True, type=str,
                        help='<Required> The output database JSON file.')
    parser.add_argument('--alignments_path', required=True, type=str,
                        help='<Required> The path to the concatenated alignments.')
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
    work_dir.mkdir(exist_ok=True, parents=True)
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
    all_marker_alignments = get_all_alignments(db, out_path.parent / "marker_genes")

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
                description=f"{strain.name}:" + "|".join(gene_ids)
            )
        )

        SeqIO.write(
            records, out_path, "fasta"
        )


def prune_db(strains: List[Strain], input_json_path: Path, output_json_path: Path, alignments_path: Path):
    # parse json entries.
    entries: Dict[str, Dict[str, Any]] = {}
    with open(input_json_path, "r") as f:
        start_entries = json.load(f)
        for entry in start_entries:
            accession = entry['accession']
            entries[accession] = entry

    # Read the alignments.
    alignments: Dict[str, np.ndarray] = {}
    for record in SeqIO.parse(alignments_path, "fasta"):
        accession = record.id
        alignments[accession] = nucleotides_to_z4(str(record.seq))

    # Cluster if hamming dist = 0.
    accessions_to_cluster: Set[str] = set(strain.id for strain in strains)
    clusters: List[List[str]] = []
    while len(accessions_to_cluster) > 0:
        cluster_rep_acc = next(iter(accessions_to_cluster))
        cluster: List[str] = []

        for other_acc in accessions_to_cluster:
            # Note: this explicitly includes cluster_rep_acc itself.
            hamming = np.sum(alignments[cluster_rep_acc] != alignments[other_acc])
            if hamming == 0:
                cluster.append(other_acc)

        accessions_to_cluster = accessions_to_cluster.difference(cluster)
        entries[cluster_rep_acc]['cluster'] = [
            "{}({})".format(
                entries[other_acc]['accession'], entries[other_acc]['strain']
            )
            for other_acc in cluster
        ]
        clusters.append(cluster)

    # Create the clustered json.
    result_entries = [
        entries[cluster[0]]
        for cluster in clusters
    ]

    with open(output_json_path, 'w') as outfile:
        json.dump(result_entries, outfile, indent=4)


def main():
    args = parse_args()

    input_json_path = Path(args.input_json)
    output_json_path = Path(args.output_json)

    input_db = JSONStrainDatabase(
        entries_file=input_json_path,
        marker_max_len=cfg.database_cfg.db_kwargs['marker_max_len'],
        force_refresh=False,
        load_full_genomes=False
    )

    alignments_path = Path(args.alignments_path)
    get_concatenated_alignments(input_db, alignments_path)
    prune_db(input_db.all_strains(), input_json_path, output_json_path, alignments_path)


if __name__ == "__main__":
    main()
