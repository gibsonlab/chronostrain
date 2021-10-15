import re
from pathlib import Path
from typing import Dict, List, Iterator
import numpy as np

from chronostrain.database import StrainDatabase
from chronostrain.model import Marker
from .sam_handler import SamFile, SamLine
from .cigar import CigarOp, CigarElement
from chronostrain.util.sequences import nucleotides_to_z4

from .alignment import SequenceReadAlignment, NucleotideDeletion, NucleotideInsertion

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


def find_start_clip(cigar_tag):
    split_cigar = re.findall('\d+|\D+', cigar_tag)
    if split_cigar[1] == 'S':
        return int(split_cigar[0])
    return 0


def parse_line_into_alignment(sam_path: Path, samline: SamLine, db: StrainDatabase) -> SequenceReadAlignment:
    cigar_els: List[CigarElement] = samline.cigar
    read_seq: str = samline.read
    read_qual: np.ndarray = samline.phred_quality

    insertions: List[NucleotideInsertion] = []
    deletions: List[NucleotideDeletion] = []

    read_seq_aln_tokens = []
    read_qual_aln_tokens = []

    # ============ Handle hard clips at the ends.
    if cigar_els[0].op == CigarOp.CLIPHARD:
        read_seq = read_seq[cigar_els[0].num:]
        read_qual = read_qual[cigar_els[0].num:]
        cigar_els = cigar_els[1:]
    if cigar_els[-1].op == CigarOp.CLIPHARD:
        read_seq = read_seq[:-cigar_els[-1].num]
        read_qual = read_qual[:-cigar_els[-1].num]
        cigar_els = cigar_els[:-1]

    # ============ Handle soft clips at the ends.
    if cigar_els[0].op == CigarOp.CLIPSOFT:
        start_clip = cigar_els[0].num
        cigar_els = cigar_els[1:]
        read_seq_aln_tokens.append(read_seq[:start_clip])
        read_qual_aln_tokens.append(read_qual[:start_clip])
    else:
        start_clip = 0

    if cigar_els[-1].op == CigarOp.CLIPSOFT:
        end_clip = cigar_els[-1].num
        cigar_els = cigar_els[:-1]
    else:
        end_clip = 0

    if start_clip > 0 and end_clip > 0:
        logger.warning("Read `{}` clipped on both ends; this might indicate repeated subsequences of different markers,"
                    "or a significant mutation (bulk insertion/deletion). Check the results at the end.")

    # ============ Handle all intermediate elements.
    current_read_idx = start_clip
    for cigar_el in cigar_els:
        if cigar_el.op == CigarOp.ALIGN:
            read_seq_aln_tokens.append(read_seq[current_read_idx:current_read_idx + cigar_el.num])
            read_qual_aln_tokens.append(read_qual[current_read_idx:current_read_idx + cigar_el.num])
            current_read_idx += cigar_el.num
        elif cigar_el.op == CigarOp.MATCH:
            read_seq_aln_tokens.append(read_seq[current_read_idx:current_read_idx + cigar_el.num])
            read_qual_aln_tokens.append(read_qual[current_read_idx:current_read_idx + cigar_el.num])
            current_read_idx += cigar_el.num
            pass
        elif cigar_el.op == CigarOp.MISMATCH:
            read_seq_aln_tokens.append(read_seq[current_read_idx:current_read_idx + cigar_el.num])
            read_qual_aln_tokens.append(read_qual[current_read_idx:current_read_idx + cigar_el.num])
            current_read_idx += cigar_el.num
            pass
        elif cigar_el.op == CigarOp.INSERTION:
            # TODO: this represents an insertion into reference marker.
            #   1) Store into the alignment instance the (position, nucleotides) pair which corresponds to
            #   this insertion.
            #   This is to be translated into (start_insert_pos, insert_len, relative_pos, nucleotide)
            #   tuples, e.g. (3, AC) means "insert AC into position 3 of marker", which translates into
            #       [(3, 2, 0, A), (3, 2, 1, C)].
            #   2) Remove from the read sequence the nucleotides inserted.
            raise NotImplementedError(
                "Line {}, Cigar `{}`: Single nucleotide insertions are not currently supported.".format(
                    samline.lineno, samline.cigar_str
                )
            )
        elif cigar_el.op == CigarOp.DELETION:
            # TODO: this represents a deletion from the reference marker.
            #   1) Store into the alignment instance the (marker_start, marker_end) pair which corresponds to the
            #   location of the deleted segment.
            #   This is to be translated into a list of (delete_pos), e.g. (3, 5) means
            #   "delete 5 chars starting at pos 3", which translates into [3, 4, 5, 6, 7].
            #   2) Add into the read sequence a special "DELETED" character/number (in z4 representation space).
            raise NotImplementedError(
                "Line {}, Cigar `{}`: Single nucleotide deletions are not currently supported.".format(
                    samline.lineno, samline.cigar_str
                )
            )
        elif cigar_el.op == CigarOp.SKIPREF:
            raise NotImplementedError(
                "Line {}, Cigar `{}`: Reference skips are not currently supported.".format(
                    samline.lineno, samline.cigar_str
                )
            )
        else:
            raise RuntimeError(
                "Cannot handle Cigar op `{}` in the middle of the aligned query. (Line {}, Cigar {})".format(
                    cigar_el.op.name,
                    samline.lineno,
                    samline.cigar_str
                )
            )

    # ============ Reattach suffix (end-of-read clipped region)
    if end_clip > 0:
        read_seq_aln_tokens.append(read_seq[-end_clip:])
        read_qual_aln_tokens.append(read_qual[-end_clip:])

    # ============ The resulting token.
    result_read_seq = nucleotides_to_z4("".join(read_seq_aln_tokens))
    result_read_qual = np.concatenate(read_qual_aln_tokens)
    assert len(result_read_seq) == len(result_read_qual)

    read_start = start_clip
    read_end = len(result_read_seq) - 1 - end_clip

    marker_start = int(samline.map_pos_str) - 1
    marker_end = marker_start + (read_end - read_start)

    # ============ Retrieve the marker.
    accession_token, name_token, id_token = samline.contig_name.split("|")
    mapped_marker = db.get_marker(
        # Assumes that the reference marker was stored automatically using Marker.to_seqrecord().
        id_token
    )

    # ============ Return the appropriate instance.
    return SequenceReadAlignment(
        samline.readname,
        sam_path,
        samline.lineno,
        result_read_seq,
        result_read_qual,
        mapped_marker,
        read_start,
        read_end,
        marker_start,
        marker_end,
        samline.is_reverse_complemented,
        insertions,
        deletions
    )


def parse_alignments(sam_file: SamFile, db: StrainDatabase) -> Iterator[SequenceReadAlignment]:
    for samline in sam_file.mapped_lines():
        try:
            yield parse_line_into_alignment(sam_file.file_path, samline, db)
        except NotImplementedError as e:
            logger.warning(str(e))


def marker_categorized_alignments(sam_file: SamFile, db: StrainDatabase) -> Dict[Marker, List[SequenceReadAlignment]]:
    marker_alignments = {
        marker: []
        for marker in db.all_markers()
    }

    for aln in parse_alignments(sam_file, db):
        marker_alignments[aln.marker].append(aln)

    return marker_alignments
