from typing import Dict, List, Tuple, Iterator
import gzip
from pathlib import Path
import pandas as pd

from intervaltree import IntervalTree
from logging import Logger

from chronostrain.util.io import open_check_compression


def find_and_resolve_overlaps(strain, refseq_index: pd.DataFrame, logger: Logger):
    """
    Detects all overlaps and handles them appropriate using helper functions (merge_markers).
    """
    t = IntervalTree()

    def add_to_tree(x, y, item):
        t[x:(y + 1)] = item

    strain_id = strain['id']
    for marker in strain['markers']:
        marker_start = marker['start']
        marker_end = marker['end']

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
            f"MERGED_{n_merged}_{strain_id}",
            logger
        )

    if n_merged > 0:
        logger.info(f"Created {n_merged} merged markers for strain {strain_id}.")


def merge_markers(
        refseq_index: pd.DataFrame,
        strain: Dict,
        start: int,
        end: int,
        to_merge: List[Dict],
        new_id: str,
        logger: Logger,
        annot_mismatch_warnings: bool = False
):
    """
    Merge a collection of markers, given a pre-computed leftmost position and rightmost position.
    Substitutes in-place the marker entries within the strain.
    """
    try:
        annot_gene_names = genomic_names_from_gff(refseq_index, strain['id'], start, end - 1)

        # was able to find GFF; now able to catch/print annotation-specific messages.
        json_gene_names = {m['name'] for m in to_merge}
        if annot_mismatch_warnings:
            for annot_gene in annot_gene_names:
                if annot_gene not in json_gene_names:
                    logger.warning(
                        "[{}] GFF annotation `{}` is different from the specified markers ({}). Check seed sequences for possibly misannotated genes.".format(
                            strain['id'],
                            annot_gene,
                            json_gene_names
                        )
                    )
        new_name = '-'.join(annot_gene_names)
    except FileNotFoundError:
        logger.debug("GFF file not found for {}.".format(strain['id']))
        new_name = "-".join(m['name'] for m in to_merge)
    ids_to_remove = set([m['id'] for m in to_merge])

    strain['markers'] = [m for m in strain['markers'] if m['id'] not in ids_to_remove] + [{
        "id": new_id,
        "name": new_name,
        "type": "subseq",
        "source": strain['id'],
        "start": start,
        "end": end,
        "strand": "+",  # by convention
        "canonical": False,
        "merged_members": [
            {'id': m['id'], 'name': m['name'], 'strand': m['strand']}
            for m in to_merge
        ]
    }]


def genomic_names_from_gff(refseq_index: pd.DataFrame, offending_strain: str, start: int, end: int) -> List[str]:
    gff_path = Path(refseq_index.loc[refseq_index['Accession'] == offending_strain, 'GFF'].item())
    hits = search_gff_annotation(gff_path, start, end, target_accession=offending_strain)
    gene_names = sorted(list(hits.keys()), key=lambda g: hits[g][1] - hits[g][0], reverse=True)  # Descending order
    return gene_names


def has_overlap(start1, end1, start2, end2):
    return (
            ((start1 <= start2) & (start2 <= end1))
            |
            ((start2 <= start1) & (start1 <= end2))
    )


def read_gff_file(file_path: Path) -> Iterator[str]:
    with open_check_compression(file_path) as handle:
        for line in handle:
            yield line


def search_gff_annotation(gff_path: Path, target_start: int, target_end: int, target_accession: str) -> Dict[str, Tuple[int, int]]:
    if gff_path.suffix == '.gz':
        f = gzip.open(gff_path, 'r')
        decode_bytes = True
    else:
        f = open(gff_path, 'r')
        decode_bytes = False

    key_to_coords = {}
    for line in f:
        if decode_bytes:
            line = line.decode('utf-8')
        tokens = line.strip().split('\t')
        acc = tokens[0]
        if acc != target_accession:
            continue

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
    f.close()
    return key_to_coords
