import re
from pathlib import Path
from typing import Tuple, Dict, List, Iterator
import numpy as np

from chronostrain.database import StrainDatabase
from chronostrain.model import Marker
from chronostrain.util.alignments import SamFile, SamLine, CigarOp
from chronostrain.util.sequences import nucleotides_to_z4, SeqType, z4_to_nucleotides

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class SequenceReadAlignment(object):
    def __init__(self,
                 read_id: str,
                 sam_path: Path,
                 sam_line_no: int,
                 read_seq: np.ndarray,
                 read_qual: np.ndarray,
                 marker: Marker,
                 read_start: int,
                 read_end: int,
                 marker_start: int,
                 marker_end: int,
                 reverse_complemented: bool):
        """
        :param read_id: The query read ID.
        :param sam_path: The SAM file that this alignment was parsed from.
        :param sam_line_no: The line number of the samhandler.
        :param read_seq: The query read sequence, minus any hard-clipped regions.
        :param read_qual: The query read quality, minus any hard-clipped regions.
        :param marker: The reference marker.
        :param read_start: The left endpoint of the read at which alignment starts; inclusive.
        :param read_end: The right endpoint of the read at which alignment starts; inclusive.
        :param marker_start: The left endpoint of the marker at which alignment starts; inclusive.
        :param marker_end: The left endpoint of the marker at which alignment starts; inclusive.
        :param reverse_complemented: Indicates whether the read sequence has been reverse-complemented from the original
            query. If so, then the quality is assumed to be reversed from the original query.
        """
        self.read_id: str = read_id
        self.sam_path: Path = sam_path
        self.sam_line_no: int = sam_line_no
        self.id: str = "{}[{},{}]".format(read_id, marker_start, marker_end)
        self.read_seq: np.ndarray = read_seq
        self.read_qual: np.ndarray = read_qual
        self.marker: Marker = marker

        assert (marker_end - marker_start) == (read_end - read_start)
        self.read_start: int = read_start
        self.read_end: int = read_end
        self.marker_start: int = marker_start
        self.marker_end: int = marker_end

        self.marker_frag: SeqType = marker.seq[marker_start:marker_end + 1]
        self.reverse_complemented: bool = reverse_complemented  # Indicates whether the read has been reverse complemented.

    @property
    def read_aligned_section(self) -> Tuple[np.ndarray, np.ndarray]:
        section = slice(self.read_start, self.read_end + 1)
        return self.read_seq[section], self.read_qual[section]

    @property
    def read_seq_nucleotide(self) -> str:
        return z4_to_nucleotides(self.read_seq)

    def __eq__(self, other: 'SequenceReadAlignment') -> bool:
        return self.id == other.id


def find_start_clip(cigar_tag):
    split_cigar = re.findall('\d+|\D+', cigar_tag)
    if split_cigar[1] == 'S':
        return int(split_cigar[0])
    return 0


def parse_line_into_alignment(sam_path: Path, samline: SamLine, db: StrainDatabase) -> SequenceReadAlignment:
    cigar_els = samline.cigar
    read_seq = samline.read
    read_qual = samline.phred_quality

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
            raise NotImplementedError(
                "Line {}, Cigar `{}`: Single nucleotide insertions are not currently supported.".format(
                    samline.lineno, samline.cigar_str
                )
            )
        elif cigar_el.op == CigarOp.DELETION:
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
        samline.is_reverse_complemented
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
