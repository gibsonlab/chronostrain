from pathlib import Path
from typing import Tuple, Dict, List, Iterator, Callable, Optional, Union
import numpy as np

from chronostrain.database import StrainDatabase, QueryNotFoundError
from chronostrain.model import Marker, SequenceRead
from chronostrain.util.sequences import SeqType, reverse_complement_seq, NucleotideDtype
from chronostrain.util.sequences import *

from ..sam import *

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
                 hard_clip_start: float,
                 hard_clip_end: float,
                 soft_clip_start: float,
                 soft_clip_end: float,
                 percent_identity: Optional[float],
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

        self.read_start: int = read_start
        self.read_end: int = read_end
        self.marker_start: int = marker_start
        self.marker_end: int = marker_end

        self.hard_clip_start = hard_clip_start  # the number of bases at the left end (5') that got hard clipped.
        self.hard_clip_end = hard_clip_end  # the number of bases at the right end (3') that got hard clipped.
        self.soft_clip_start = soft_clip_start  # the number of bases at the right end (5') that got soft clipped.
        self.soft_clip_end = soft_clip_end  # the number of bases at the right end (3') that got soft clipped.

        self.percent_identity: Union[None, float] = percent_identity

        # Indicates whether the read has been reverse complemented.
        self.reverse_complemented: bool = reverse_complemented

    @property
    def is_edge_mapped(self) -> bool:
        if self.marker_start == 0:
            return self.soft_clip_start > 0 or self.hard_clip_start > 0
        elif self.marker_end == len(self.marker) - 1:
            return self.soft_clip_end > 0 or self.hard_clip_end > 0
        return False

    @property
    def read_aligned_section(self) -> Tuple[SeqType, SeqType]:
        """
        Returns the section of the read corresponding to matches/mismatches.
        Specifically removes clipped edges. Optionally deletes insertions and/or deletions.
        """
        section = slice(self.read_start, self.read_end + 1)
        return self.read.seq[section], self.read.quality[section]

    def read_insertion_locs(self) -> np.ndarray:
        insertion_locs = np.equal(self.aln_matrix[0], nucleotide_GAP_z4)
        return insertion_locs[self.aln_matrix[1] != nucleotide_GAP_z4]

    def marker_deletion_locs(self) -> np.ndarray:
        deletion_locs = np.equal(self.aln_matrix[1], nucleotide_GAP_z4)
        return deletion_locs[self.aln_matrix[0] != nucleotide_GAP_z4]

    def __eq__(self, other: 'SequenceReadPairwiseAlignment') -> bool:
        return self.unique_id == other.unique_id


def parse_line_into_alignment(sam_path: Path,
                              samline: SamLine,
                              db: StrainDatabase,
                              read_getter: Optional[Callable[[str], SequenceRead]] = None) -> SequenceReadPairwiseAlignment:
    """
    Parse a given SamLine (excluding metadata) into an alignment instance.
    """
    # ============ Retrieve the read.
    if read_getter is not None:
        read = read_getter(samline.readname)
    else:
        if samline.is_reverse_complemented:
            read = SequenceRead(
                read_id=samline.readname,
                seq=reverse_complement_seq(nucleotides_to_z4(samline.read[::-1])),
                quality=samline.phred_quality[::-1],
                metadata=f"Sam_parsed(f={str(sam_path)},L={samline.lineno},revcomp)"
            )
        else:
            read = SequenceRead(
                read_id=samline.readname,
                seq=nucleotides_to_z4(samline.read),
                quality=samline.phred_quality,
                metadata=f"Sam_parsed(f={str(sam_path)},L={samline.lineno})"
            )

    # ============ Retrieve the marker.
    accession_token, name_token, id_token = samline.contig_name.strip().split(" ")[0].split("|")
    try:
        marker = db.get_marker(
            # Assumes that the reference marker was stored automatically using Marker.to_seqrecord().
            id_token
        )
    except QueryNotFoundError as e:
        logger.error("Encountered bad marker ID from token {}. File: {}, Line: {}.".format(
            samline.contig_name,
            str(sam_path),
            samline.lineno
        ))
        raise e

    # ============ Parse the cigar string to generate alignments.
    cigar_els: List[CigarElement] = samline.cigar

    if samline.is_reverse_complemented:
        read_seq = reverse_complement_seq(read.seq)
    else:
        read_seq = read.seq

    read_tokens = []
    marker_tokens = []
    hard_clip_start = 0
    hard_clip_end = 0
    soft_clip_start = 0
    soft_clip_end = 0

    # ============ Handle hard clips at the ends (always comes before the soft clips).
    if cigar_els[0].op == CigarOp.CLIPHARD:
        hard_clip_start = cigar_els[0].num
        cigar_els = cigar_els[1:]
    if cigar_els[-1].op == CigarOp.CLIPHARD:
        hard_clip_end = cigar_els[-1].num
        cigar_els = cigar_els[:-1]

    # ============ Handle soft clips at the ends.
    if cigar_els[0].op == CigarOp.CLIPSOFT:
        soft_clip_start = cigar_els[0].num
        cigar_els = cigar_els[1:]
    if cigar_els[-1].op == CigarOp.CLIPSOFT:
        soft_clip_end = cigar_els[-1].num
        cigar_els = cigar_els[:-1]

    marker_start = int(samline.map_pos_str) - 1
    read_start = soft_clip_start + hard_clip_start

    # ============ Handle all intermediate elements.
    current_read_idx = read_start
    current_marker_idx = marker_start
    for cigar_el in cigar_els:
        if cigar_el.op == CigarOp.ALIGN or cigar_el.op == CigarOp.MATCH or cigar_el.op == CigarOp.MISMATCH:
            # Consume both query and marker.
            read_tokens.append(read_seq[current_read_idx:current_read_idx + cigar_el.num])
            marker_tokens.append(marker.seq[current_marker_idx:current_marker_idx + cigar_el.num])
            current_read_idx += cigar_el.num
            current_marker_idx += cigar_el.num
        elif cigar_el.op == CigarOp.INSERTION:
            # Consume query but not marker.
            read_tokens.append(read_seq[current_read_idx:current_read_idx + cigar_el.num])
            marker_tokens.append(nucleotide_GAP_z4 * np.ones(cigar_el.num, dtype=NucleotideDtype))
            current_read_idx += cigar_el.num
        elif cigar_el.op == CigarOp.DELETION:
            # consume marker but not query.
            read_tokens.append(nucleotide_GAP_z4 * np.ones(cigar_el.num, dtype=NucleotideDtype))
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
        np.concatenate(read_tokens)
    ], axis=0)  # matrix of characters and gaps showing the alignment. (third row shows quality scores of each base.)

    frag = alignment[0]
    fragment = frag[frag != nucleotide_GAP_z4]

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
        hard_clip_start,
        hard_clip_end,
        soft_clip_start,
        soft_clip_end,
        samline.percent_identity,
        samline.is_reverse_complemented
    )


def parse_alignments(sam_file: SamFile,
                     db: StrainDatabase,
                     read_getter: Optional[Callable[[str], SequenceRead]] = None) -> Iterator[SequenceReadPairwiseAlignment]:
    """
    A basic function which parses a SamFile instance and outputs a generator over alignments.
    :param sam_file: The Sam file to parse.
    :param db: The database to create.
    :param read_getter:
    """
    for samline in sam_file.mapped_lines():
        try:
            yield parse_line_into_alignment(sam_file.file_path, samline, db, read_getter)
        except NotImplementedError as e:
            logger.warning(str(e))


def marker_categorized_alignments(sam_file: SamFile,
                                  db: StrainDatabase,
                                  read_getter: Callable[[str], SequenceRead],
                                  ignore_edge_mapped_reads: bool = True
                                  ) -> Dict[Marker, List[SequenceReadPairwiseAlignment]]:
    """
    Parses the input SamFile instance into a dictionary, mapping each marker to alignments that map to
    it.
    """
    marker_alignments = {
        marker: []
        for marker in db.all_markers()
    }

    for aln in parse_alignments(sam_file,
                                db,
                                read_getter=read_getter):
        if ignore_edge_mapped_reads and aln.is_edge_mapped:
            logger.debug(f"Ignoring alignment of read {aln.read.id} to marker {aln.marker.id} "
                         f"({aln.sam_path.name}, Line {aln.sam_line_no}), which is edge-mapped.")
            continue
        marker_alignments[aln.marker].append(aln)

    return marker_alignments
