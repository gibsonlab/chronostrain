from typing import List, Tuple, Dict
import argparse
from pathlib import Path
import json
import pandas as pd

from intervaltree import IntervalTree
from tqdm import tqdm
from Bio import SeqIO

from chronostrain.logging import create_logger
logger = create_logger("chronostrain.validate_db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_json", required=True, type=str,
                        help='<Required> The target input database (JSON-formatted specification) to resolve.')
    parser.add_argument("-o", "--output_json", required=True, type=str,
                        help='<Required> The target output database file path')
    parser.add_argument("-r", "--refseq_index", required=True, type=str,
                        help='<Required> The index for the RefSeq collection.')
    return parser.parse_args()


def flip_strand(start, end, total) -> Tuple[int, int]:
    new_start = total - end + 1
    new_end = total - start + 1
    return new_start, new_end


def has_overlap(start1, end1, start2, end2):
    return (
            ((start1 <= start2) & (start2 <= end1))
            |
            ((start2 <= start1) & (start1 <= end2))
    )


def search_gff_annotation(gff_path: Path, target_start: int, target_end: int) -> Dict[str, Tuple[int, int]]:
    with open(gff_path, "rt") as f:
        key_to_coords = {}
        for line in f:
            tokens = line.strip().split('\t')
            datum = {}
            for entry in tokens[8].split(';'):
                k, v = entry.split('=')
                datum[k] = v

            if datum['gbkey'] != 'Gene':
                continue

            item_start = int(tokens[3])
            item_end = int(tokens[4])
            try:
                item_key = datum['gene']
            except KeyError:
                item_key = datum['Name']

            if has_overlap(target_start, target_end, item_start, item_end):
                overlap_start = max(item_start, target_start)
                overlap_end = min(item_end, target_end)
                key_to_coords[item_key] = (overlap_start, overlap_end)
        return key_to_coords


def resolve_overlap(refseq_index: pd.DataFrame, offending_strain: str, start: int, end: int) -> str:
    chromosome_path = Path(refseq_index.loc[refseq_index['Accession'] == offending_strain, 'SeqPath'].item())
    assembly_id = refseq_index.loc[refseq_index['Accession'] == offending_strain, 'Assembly'].item()
    gff_files = list(chromosome_path.parent.glob(f"{assembly_id}_*_genomic.chrom.gff"))
    if len(gff_files) == 0:
        raise RuntimeError(f"No annotations found for {offending_strain} (assembly: {assembly_id}).")
    elif len(gff_files) > 1:
        raise RuntimeError(f"Multiple annotation files found for {offending_strain} (assembly: {assembly_id}).")

    gff_file = gff_files[0]
    hits = search_gff_annotation(gff_file, start, end)
    gene_names = sorted(list(hits.keys()), key=lambda g: hits[g][1] - hits[g][0], reverse=True)  # Descending order
    new_name = '-'.join(gene_names)
    return new_name


def merge_markers(refseq_index: pd.DataFrame, strain: Dict, start: int, end: int, to_merge: List[Dict], new_id: str):
    """
    Merge a collection of markers, given a pre-computed leftmost position and rightmost position.
    Substitutes in-place the marker entries within the strain.
    """
    new_name = resolve_overlap(refseq_index, strain['id'], start, end - 1)
    ids_to_remove = set([m['id'] for m in to_merge])

    strain['markers'] = [m for m in strain['markers'] if m['id'] not in ids_to_remove] + [{
        "id": new_id,
        "name": new_name,
        "type": "subseq",
        "source": strain['id'],
        "start": start,
        "end": end,
        "strand": "+",
        "canonical": False,
        "members": [m['id'] for m in to_merge],
        "member_names": "+".join(sorted(m['name'] for m in to_merge))
    }]


def find_and_resolve_overlaps(strain, refseq_index: pd.DataFrame):
    """
    Detects all overlaps and handles them appropriate using helper functions (merge_markers).
    """
    t = IntervalTree()

    def add_to_tree(x, y, item):
        t[x:(y + 1)] = item

    strain_id = strain['id']
    chromosome_path = refseq_index.loc[refseq_index['Accession'] == strain_id, 'SeqPath'].item()
    record = SeqIO.read(chromosome_path, format='fasta')
    genome_len = len(record)

    for marker in strain['markers']:
        strand = marker['strand']
        strand_start = marker['start']
        strand_end = marker['end']
        if strand == '+':
            marker_start, marker_end = strand_start, strand_end
        elif strand == '-':
            marker_start, marker_end = flip_strand(strand_start, strand_end, genome_len)
        else:
            raise RuntimeError(f"Unexpected strand `{strand}`")

        add_to_tree(marker_start, marker_end, marker)

    def _reducer(cur, x):
        cur.append(x)
        return cur

    t.merge_overlaps(
        data_reducer=_reducer,
        data_initializer=[],
        strict=False
    )

    n_merged = 0
    for interval in t:
        if len(interval.data) <= 1:
            continue

        n_merged += 1
        merge_markers(
            refseq_index,
            strain,
            interval.begin,
            interval.end - 1,
            interval.data,
            f"MERGED_{n_merged}_{strain_id}"
        )

    if n_merged > 0:
        logger.info(f"Created {n_merged} merged markers for strain {strain_id}.")
        for marker in strain['markers']:
            if 'members' in marker:
                logger.debug("{} -> {}".format(
                    ' + '.join(marker['members']),
                    marker.name
                ))


def main():
    args = parse_args()

    input_json_path = Path(args.input_json)
    output_json_path = Path(args.output_json)
    refseq_index = pd.read_csv(args.refseq_index, sep='\t')

    with open(input_json_path, "r") as f:
        db_json = json.load(f)

    for strain in tqdm(db_json):
        find_and_resolve_overlaps(strain, refseq_index)

    with open(output_json_path, 'w') as o:
        json.dump(db_json, o, indent=4)


if __name__ == "__main__":
    main()
