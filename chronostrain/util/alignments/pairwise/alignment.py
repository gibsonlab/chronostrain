from pathlib import Path
from typing import Tuple, Dict, List, Iterator, Callable, Optional, Union
import numpy as np

from chronostrain.database import StrainDatabase, QueryNotFoundError
from chronostrain.model import Marker, SequenceRead
from chronostrain.util.sequences import *

from ..sam import *

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class SequenceReadPairwiseAlignment(object):
    def __init__(self,
                 read: SequenceRead,
                 marker: Marker,
                 aln_matrix: np.ndarray,
                 sam_path: Path,
                 sam_line_no: int,
                 read_start: int,
                 read_end: int,
                 marker_start: int,
                 marker_end: int,
                 hard_clip_start: int,
                 hard_clip_end: int,
                 soft_clip_start: int,
                 soft_clip_end: int,
                 num_aligned_bases: Optional[int],
                 num_mismatches: Optional[int],
                 reverse_complemented: bool):
        """
        :param read: The SequenceRead instance.
        :param marker: The reference marker.
        :param fragment: The sequence corresponding to the gapless marker fragment that the read mapped to.
        :param aln_matrix: a 2xL matrix consisting of two rows: the gapped marker fragment, the gapped read seq.
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

        self.aln_matrix: np.ndarray = aln_matrix

        self.sam_path: Path = sam_path
        self.sam_line_no: int = sam_line_no
        self.unique_id: str = "{}[{},{}]".format(read.id, marker_start, marker_end)

        self.read_start: int = read_start
        self.read_end: int = read_end
        self.marker_start: int = marker_start
        self.marker_end: int = marker_end

        self.hard_clip_start: int = hard_clip_start  # the number of bases at the left end (5') that got hard clipped.
        self.hard_clip_end: int = hard_clip_end  # the number of bases at the right end (3') that got hard clipped.
        self.soft_clip_start: int = soft_clip_start  # the number of bases at the right end (5') that got soft clipped.
        self.soft_clip_end: int = soft_clip_end  # the number of bases at the right end (3') that got soft clipped.

        self.num_aligned_bases: Union[None, int] = num_aligned_bases
        self.num_mismatches: Union[None, int] = num_mismatches

        # Indicates whether the read has been reverse complemented.
        self.reverse_complemented: bool = reverse_complemented

    @property
    def marker_frag(self) -> SeqType:
        frag_with_gaps = self.aln_matrix[0]
        return frag_with_gaps[frag_with_gaps != nucleotide_GAP_z4]

    @property
    def is_clipped(self) -> bool:
        return self.soft_clip_start > 0 or self.hard_clip_start > 0 or self.soft_clip_end > 0 or self.hard_clip_end > 0

    @property
    def is_edge_mapped(self) -> bool:
        return (self.marker_start == 0) or (self.marker_end == len(self.marker) - 1)

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
                seq=reverse_complement_seq(samline.read_seq),
                quality=samline.read_phred[::-1],
                metadata=f"Sam_parsed(f={str(sam_path)},L={samline.lineno},revcomp)"
            )
        else:
            read = SequenceRead(
                read_id=samline.readname,
                seq=samline.read_seq,
                quality=samline.read_phred,
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

    marker_start = samline.contig_map_idx  # first included index
    read_start = soft_clip_start + hard_clip_start  # first included index

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
                f"Line {samline.lineno}, Cigar `{samline.cigar}`: Reference skip is only expected for "
                "mRNA-to-genome alignment. Check the alignment tool used."
            )
        else:
            raise RuntimeError(
                f"Cannot handle Cigar op `{cigar_el.op.name}` in the middle of the aligned query. "
                f"(Line {samline.lineno}, Cigar {samline.cigar})"
            )

    read_end = current_read_idx - 1  # last included index
    marker_end = current_marker_idx - 1  # last included index
    assert read_end >= read_start
    assert marker_end >= marker_start

    marker_seq = np.concatenate(marker_tokens)
    read_seq = np.concatenate(read_tokens)

    """
    Matrix of characters and gaps showing the alignment.
    """
    alignment = np.stack([marker_seq, read_seq], axis=0)

    # ============ Return the appropriate instance.
    return SequenceReadPairwiseAlignment(
        read,
        marker,
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
        samline.num_aligned_bases,
        samline.num_mismatches,
        samline.is_reverse_complemented
    )


def reattach_clipped_bases_to_aln(aln: SequenceReadPairwiseAlignment):
    """
    For the special case where bases have been clipped but the mapping is not at the edge, append the rest of
    the fragment.
    """
    is_left_edge_mapped = (aln.marker_start == 0)
    is_right_edge_mapped = (aln.marker_end == len(aln.marker) - 1)
    n_start_clip = min(aln.soft_clip_start + aln.hard_clip_start, aln.marker_start)

    aligned_marker_seq = aln.aln_matrix[0]
    aligned_read_seq = aln.aln_matrix[1]

    if aln.reverse_complemented:
        read_seq = reverse_complement_seq(aln.read.seq)
    else:
        read_seq = aln.read.seq

    if not is_left_edge_mapped and n_start_clip > 0:
        marker_prefix = aln.marker.seq[(aln.marker_start - n_start_clip):aln.marker_start]
        aligned_marker_seq = np.concatenate([marker_prefix, aligned_marker_seq])

        read_prefix = read_seq[(aln.read_start - n_start_clip):aln.read_start]
        aligned_read_seq = np.concatenate([read_prefix, aligned_read_seq])
        if aln.hard_clip_start > 0:
            aln.hard_clip_start -= n_start_clip
        elif aln.soft_clip_start > 0:
            aln.soft_clip_start -= n_start_clip
        aln.marker_start -= n_start_clip
        aln.read_start -= n_start_clip

    n_end_clip = min(aln.soft_clip_end + aln.hard_clip_end, len(aln.marker) - aln.marker_end - 1)
    if not is_right_edge_mapped and n_end_clip > 0:
        marker_suffix = aln.marker.seq[(aln.marker_end + 1):(aln.marker_end + 1 + n_end_clip)]
        aligned_marker_seq = np.concatenate([aligned_marker_seq, marker_suffix])

        read_suffix = read_seq[(aln.read_end + 1):(aln.read_end + 1 + n_end_clip)]
        aligned_read_seq = np.concatenate([aligned_read_seq, read_suffix])
        if aln.hard_clip_end > 0:
            aln.hard_clip_end -= n_end_clip
        elif aln.soft_clip_end > 0:
            aln.soft_clip_end -= n_end_clip
        aln.marker_end += n_end_clip
        aln.read_end += n_end_clip

    aln.aln_matrix = np.stack([aligned_marker_seq, aligned_read_seq], axis=0)


def parse_alignments(sam_file: SamFile,
                     db: StrainDatabase,
                     read_getter: Optional[Callable[[str], SequenceRead]] = None,
                     reattach_clipped_bases: bool = False,
                     min_hit_ratio: float = 0.75) -> Iterator[SequenceReadPairwiseAlignment]:
    """
    A basic function which parses a SamFile instance and outputs a generator over alignments.
    :param sam_file: The Sam file to parse.
    :param db: The database to create.
    :param read_getter:
    """
    for samline in sam_file.mapped_lines():
        try:
            read = read_getter(samline.readname)
            if len(samline.read_seq) / len(read) <= min_hit_ratio:
                pass

            aln = parse_line_into_alignment(sam_file.file_path,
                                            samline,
                                            db,
                                            read_getter)

            if reattach_clipped_bases:
                reattach_clipped_bases_to_aln(aln)

            yield aln
        except NotImplementedError as e:
            logger.warning(str(e))


def marker_categorized_alignments(sam_file: SamFile,
                                  db: StrainDatabase,
                                  read_getter: Callable[[str], SequenceRead],
                                  reattach_clipped_bases: bool = False,
                                  min_hit_ratio: float = 0.75
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
                                read_getter=read_getter,
                                reattach_clipped_bases=reattach_clipped_bases,
                                min_hit_ratio=min_hit_ratio):
        marker_alignments[aln.marker].append(aln)

    return marker_alignments
