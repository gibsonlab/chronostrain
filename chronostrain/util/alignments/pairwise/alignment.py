from pathlib import Path
from typing import Tuple, Dict, List, Iterator, Callable
import numpy as np

from chronostrain.database import StrainDatabase
from chronostrain.model import Marker, SequenceRead
from chronostrain.util.sequences import SeqType, reverse_complement_seq
from chronostrain.util.sequences import nucleotide_GAP_z4 as GapChar

from .sam_handler import SamFile, SamLine
from .cigar import CigarOp, CigarElement

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class SequenceReadPairwiseAlignment(object):
    def __init__(self,
                 read: SequenceRead,
                 marker: Marker,
                 fragment: SeqType,
                 aln_matrix: np.ndarray,
                 sam_path: Path,
                 sam_line_no: int,
                 read_start: int,
                 read_end: int,
                 marker_start: int,
                 marker_end: int,
                 reverse_complemented: bool):
        """
        :param read: The SequenceRead instance.
        :param marker: The reference marker.
        :param fragment: The sequence corresponding to the gapless marker fragment that the read mapped to.
        :param aln_matrix: a 3xL matrix consisting of three rows: the gapped marker fragment, the gapped read seq,
            and the gapped quality score vector (mirroring the second row).
        :param sam_path: The SAM file that this alignment was parsed from.
        :param sam_line_no: The line number of the samhandler.
        :param read_start: The left endpoint of the read at which alignment starts; inclusive.
        :param read_end: The right endpoint of the read at which alignment starts; inclusive.
        :param marker_start: The left endpoint of the marker at which alignment starts; inclusive.
        :param marker_end: The left endpoint of the marker at which alignment starts; inclusive.
        :param reverse_complemented: Indicates whether the read sequence (in aln_matrix) has been reverse-complemented
            from the original query. If so, then the quality is assumed to be reversed from the original query.
        """
        self.read: SequenceRead = read
        self.marker: Marker = marker
        self.marker_frag: SeqType = fragment

        self.aln_matrix: np.ndarray = aln_matrix

        self.sam_path: Path = sam_path
        self.sam_line_no: int = sam_line_no
        self.unique_id: str = "{}[{},{}]".format(read.id, marker_start, marker_end)

        assert (marker_end - marker_start) == (read_end - read_start)
        self.read_start: int = read_start
        self.read_end: int = read_end
        self.marker_start: int = marker_start
        self.marker_end: int = marker_end

        # Indicates whether the read has been reverse complemented.
        self.reverse_complemented: bool = reverse_complemented

    @property
    def read_aligned_section(self) -> Tuple[SeqType, SeqType]:
        """
        Returns the section of the read corresponding to matches/mismatches.
        Specifically removes clipped edges. Optionally deletes insertions and/or deletions.
        """
        section = slice(self.read_start, self.read_end + 1)
        return self.read.seq[section], self.read.quality[section]

    def read_insertion_locs(self) -> np.ndarray:
        insertion_locs = np.equal(self.aln_matrix[0], GapChar)
        return insertion_locs[self.aln_matrix[1] != GapChar]

    def marker_deletion_locs(self) -> np.ndarray:
        deletion_locs = np.equal(self.aln_matrix[1], GapChar)
        return deletion_locs[self.aln_matrix[0] != GapChar]

    def __eq__(self, other: 'SequenceReadPairwiseAlignment') -> bool:
        return self.unique_id == other.unique_id


def parse_line_into_alignment(sam_path: Path,
                              samline: SamLine,
                              db: StrainDatabase,
                              read_getter: Callable[[str], SequenceRead]) -> SequenceReadPairwiseAlignment:
    """
    Parse a given SamLine (excluding metadata) into an alignment instance.
    """
    # ============ Retrieve the read.
    read = read_getter(samline.readname)

    # ============ Retrieve the marker.
    accession_token, name_token, id_token = samline.contig_name.split("|")
    marker = db.get_marker(
        # Assumes that the reference marker was stored automatically using Marker.to_seqrecord().
        id_token
    )

    # ============ Parse the cigar string to generate alignments.
    cigar_els: List[CigarElement] = samline.cigar

    if samline.is_reverse_complemented:
        read_seq = reverse_complement_seq(read.seq)
        qual_seq = read.quality[::-1]
    else:
        read_seq = read.seq
        qual_seq = read.quality

    read_tokens = []
    qual_tokens = []
    marker_tokens = []

    # ============ Handle hard clips at the ends.
    if cigar_els[0].op == CigarOp.CLIPHARD:
        cigar_els = cigar_els[1:]
    if cigar_els[-1].op == CigarOp.CLIPHARD:
        cigar_els = cigar_els[:-1]

    # ============ Handle soft clips at the ends.
    if cigar_els[0].op == CigarOp.CLIPSOFT:
        start_clip = cigar_els[0].num
        cigar_els = cigar_els[1:]
    else:
        start_clip = 0

    if cigar_els[-1].op == CigarOp.CLIPSOFT:
        end_clip = cigar_els[-1].num
        cigar_els = cigar_els[:-1]
    else:
        end_clip = 0

    if start_clip > 0 and end_clip > 0:
        logger.warning("Read `{}` clipped on both ends; this might indicate repeated subsequences of different "
                       "markers, or a significant mutation (bulk insertion/deletion). Check the results at the end.")

    marker_start = int(samline.map_pos_str) - 1
    read_start = start_clip

    # ============ Handle all intermediate elements.
    current_read_idx = read_start
    current_marker_idx = marker_start
    for cigar_el in cigar_els:
        if cigar_el.op == CigarOp.ALIGN or cigar_el.op == CigarOp.MATCH or cigar_el.op == CigarOp.MISMATCH:
            # Consume both query and marker.
            read_tokens.append(read_seq[current_read_idx:current_read_idx + cigar_el.num])
            qual_tokens.append(qual_seq[current_read_idx:current_read_idx + cigar_el.num])
            marker_tokens.append(marker.seq[current_marker_idx:current_marker_idx + cigar_el.num])
            current_read_idx += cigar_el.num
            current_marker_idx += cigar_el.num
        elif cigar_el.op == CigarOp.INSERTION:
            # Consume query but not marker.
            read_tokens.append(read_seq[current_read_idx:current_read_idx + cigar_el.num])
            qual_tokens.append(qual_seq[current_read_idx:current_read_idx + cigar_el.num])
            marker_tokens.append(GapChar * np.ones(cigar_el.num))
            current_read_idx += cigar_el.num
        elif cigar_el.op == CigarOp.DELETION:
            # consume marker but not query.
            read_tokens.append(GapChar * np.ones(cigar_el.num))
            qual_tokens.append(np.zeros(cigar_el.num))
            marker_tokens.append(marker.seq[current_marker_idx:current_marker_idx + cigar_el.num])
            current_marker_idx += cigar_el.num
        elif cigar_el.op == CigarOp.SKIPREF:
            raise ValueError(
                f"Line {samline.lineno}, Cigar `{samline.cigar_str}`: Reference skip is only expected for "
                "mRNA-to-genome alignment. Check the alignment tool used."
            )
        else:
            raise RuntimeError(
                f"Cannot handle Cigar op `{cigar_el.op.name}` in the middle of the aligned query. "
                f"(Line {samline.lineno}, Cigar {samline.cigar_str})"
            )

    read_end = current_read_idx - 1
    marker_end = current_marker_idx - 1
    assert read_end >= read_start
    assert marker_end >= marker_start

    alignment = np.stack([
        np.concatenate(marker_tokens),
        np.concatenate(read_tokens),
        np.concatenate(qual_tokens)
    ], axis=0)  # matrix of characters and gaps showing the alignment. (third row shows quality scores of each base.)

    frag = alignment[0]
    fragment = frag[frag != GapChar]

    # ============ Return the appropriate instance.
    return SequenceReadPairwiseAlignment(
        read,
        marker,
        fragment,
        alignment,
        sam_path,
        samline.lineno,
        read_start,
        read_end,
        marker_start,
        marker_end,
        samline.is_reverse_complemented
    )


def parse_alignments(sam_file: SamFile,
                     db: StrainDatabase,
                     read_getter: Callable[[str], SequenceRead]) -> Iterator[SequenceReadPairwiseAlignment]:
    """
    A basic function which parses a SamFile instance and outputs a generator over alignments.
    """
    for samline in sam_file.mapped_lines():
        try:
            yield parse_line_into_alignment(sam_file.file_path, samline, db, read_getter)
        except NotImplementedError as e:
            logger.warning(str(e))


def marker_categorized_alignments(sam_file: SamFile,
                                  db: StrainDatabase,
                                  read_getter: Callable[[str], SequenceRead]
                                  ) -> Dict[Marker, List[SequenceReadPairwiseAlignment]]:
    """
    Parses the input SamFile instance into a dictionary, mapping each marker to alignments that map to
    it.
    """
    marker_alignments = {
        marker: []
        for marker in db.all_markers()
    }

    for aln in parse_alignments(sam_file, db, read_getter):
        marker_alignments[aln.marker].append(aln)

    return marker_alignments
