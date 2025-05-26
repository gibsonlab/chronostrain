from pathlib import Path
from typing import Tuple, List, Iterator, Callable, Optional
import numpy as np

from chronostrain.database import StrainDatabase, QueryNotFoundError
from chronostrain.model import Marker, SequenceRead
from chronostrain.util.sequences import AllocatedSequence, bytes_GAP, NucleotideDtype

from ..sam import *

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class SequenceReadPairwiseAlignment(object):
    def __init__(self,
                 read: SequenceRead,
                 marker: Marker,
                 marker_frag_aln: np.ndarray,
                 read_seq_aln: np.ndarray,
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
        :param marker_frag_aln: a length L array containing the marker alignment fragment.
        :param read_seq_aln: a length L array containing the read alignment.
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

        self.marker_frag_aln_with_gaps = marker_frag_aln
        self.read_aln_with_gaps = read_seq_aln
        assert len(marker_frag_aln.shape) == 1
        assert len(read_seq_aln.shape) == 1
        assert marker_frag_aln.shape[0] == read_seq_aln.shape[0]

        self.sam_path: Path = sam_path
        self.sam_line_no: int = sam_line_no
        self.unique_id: str = "{}[{},{}]".format(read.id, marker_start, marker_end)

        self.read_start: int = read_start
        self.read_end: int = read_end
        self.marker_start: int = marker_start
        self.marker_end: int = marker_end

        # Indicates whether the read has been reverse complemented.
        self.reverse_complemented: bool = reverse_complemented

    @property
    def marker_frag(self) -> AllocatedSequence:
        return AllocatedSequence(self.marker_frag_aln_with_gaps[self.marker_frag_aln_with_gaps != bytes_GAP])

    @property
    def is_edge_mapped(self) -> bool:
        return (self.marker_start == 0) or (self.marker_end == len(self.marker) - 1)

    @property
    def read_aligned_section(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the section of the read corresponding to the alignment.
        """
        section = slice(self.read_start, self.read_end + 1)
        return self.read.seq.bytes()[section], self.read.quality[section]

    @property
    def num_mismatches(self) -> int:
        m_bases = np.not_equal(self.marker_frag_aln_with_gaps, bytes_GAP)
        r_bases = np.not_equal(self.read_aln_with_gaps, bytes_GAP)
        mismatches = np.not_equal(self.marker_frag_aln_with_gaps, self.read_aln_with_gaps)
        return np.sum(
            np.logical_and(
                m_bases & r_bases,
                mismatches
            )
        ).item()

    @property
    def num_matches(self) -> int:
        return np.sum(
            np.equal(self.marker_frag_aln_with_gaps, self.read_aln_with_gaps)
        ).item()

    def read_insertion_locs(self) -> np.ndarray:
        insertion_locs = np.equal(self.marker_frag_aln_with_gaps, bytes_GAP)
        return insertion_locs[self.read_aln_with_gaps != bytes_GAP]

    def marker_deletion_locs(self) -> np.ndarray:
        deletion_locs = np.equal(self.read_aln_with_gaps, bytes_GAP)
        return deletion_locs[self.marker_frag_aln_with_gaps != bytes_GAP]

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
                seq=samline.read_seq.revcomp_seq(),
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
    name_token, id_token = samline.ref_name.strip().split(" ")[0].split("|")
    try:
        marker = db.get_marker(
            # Assumes that the reference marker was stored automatically using Marker.to_seqrecord().
            id_token
        )
    except QueryNotFoundError as e:
        logger.error("Encountered bad marker ID from token {}. File: {}, Line: {}.".format(
            samline.ref_name,
            str(sam_path),
            samline.lineno
        ))
        raise e

    # ============ Handle the cigar string; check for clipped bases.
    cigar_els: List[CigarElement] = samline.cigar

    hard_clip_start = 0
    hard_clip_end = 0
    soft_clip_start = 0
    soft_clip_end = 0

    # Count hard clips at the ends (always comes before the soft clips).
    if cigar_els[0].op == CigarOp.CLIPHARD:
        hard_clip_start = cigar_els[0].num
        cigar_els = cigar_els[1:]
    if cigar_els[-1].op == CigarOp.CLIPHARD:
        hard_clip_end = cigar_els[-1].num
        cigar_els = cigar_els[:-1]

    # Count soft clips at the ends.
    if cigar_els[0].op == CigarOp.CLIPSOFT:
        soft_clip_start = cigar_els[0].num
        cigar_els = cigar_els[1:]
    if cigar_els[-1].op == CigarOp.CLIPSOFT:
        soft_clip_end = cigar_els[-1].num
        cigar_els = cigar_els[:-1]

    # Count number of deletions.
    n_insertions = sum(
        cigar_el.num
        for cigar_el in cigar_els
        if cigar_el.op == CigarOp.INSERTION
    )
    n_deletions = sum(
        cigar_el.num
        for cigar_el in cigar_els
        if cigar_el.op == CigarOp.DELETION
    )

    n_clip_start = soft_clip_start + hard_clip_start
    n_clip_end = soft_clip_end + hard_clip_end
    # marker_start = samline.contig_map_idx  # first included index

    # If possible, reattach clipped bases and convert local alignment to global.
    # Note that these are all 0-indexed coordinates, end coord is exclusive, e.g. x[start:end]
    global_ref_start_pos = max(samline.ref_start - n_clip_start, 0)
    local_ref_start_pos = samline.ref_start
    global_ref_end_pos = min(samline.ref_end + n_clip_end, len(marker))
    local_ref_end_pos = samline.ref_end

    start_reattach_bases = local_ref_start_pos - global_ref_start_pos
    end_reattach_bases = global_ref_end_pos - local_ref_end_pos
    marker_frag_len = global_ref_end_pos - global_ref_start_pos

    # Coordinates on the read
    local_read_start_pos = n_clip_start
    global_read_start_pos = local_read_start_pos - start_reattach_bases
    local_read_end_pos = len(read) - n_clip_end
    global_read_end_pos = local_read_end_pos + end_reattach_bases

    alignment_length = marker_frag_len + n_insertions
    assert alignment_length == global_read_end_pos - global_read_start_pos + n_deletions
    marker_frag_aln = np.full(alignment_length, fill_value=255, dtype=NucleotideDtype)
    read_aln = np.full(alignment_length, fill_value=255, dtype=NucleotideDtype)

    # Reattach clipped bases.
    if samline.is_reverse_complemented:
        read_seq = read.seq.revcomp_bytes()
    else:
        read_seq = read.seq.bytes()
    if start_reattach_bases > 0:
        marker_frag_aln[:start_reattach_bases] = marker.seq.bytes()[global_ref_start_pos:local_ref_start_pos]
        read_aln[:start_reattach_bases] = read_seq[global_read_start_pos:local_read_start_pos]
    if end_reattach_bases > 0:
        marker_frag_aln[-end_reattach_bases:] = marker.seq.bytes()[local_ref_end_pos:global_ref_end_pos]
        read_aln[-end_reattach_bases:] = read_seq[local_read_end_pos:global_read_end_pos]

    # Iterate through cigar elements.
    cur_aln_pos = start_reattach_bases
    cur_marker_pos = local_ref_start_pos
    cur_read_pos = local_read_start_pos
    for cigar_el in cigar_els:
        n = cigar_el.num
        if cigar_el.op == CigarOp.ALIGN or cigar_el.op == CigarOp.MATCH or cigar_el.op == CigarOp.MISMATCH:
            # Consume both query and marker.
            marker_frag_aln[cur_aln_pos:cur_aln_pos + n] = marker.seq.bytes()[cur_marker_pos:cur_marker_pos + n]
            read_aln[cur_aln_pos:cur_aln_pos + n] = read_seq[cur_read_pos:cur_read_pos + n]
            cur_marker_pos += n
            cur_read_pos += n
        elif cigar_el.op == CigarOp.INSERTION:
            # Consume query but not marker.
            marker_frag_aln[cur_aln_pos:cur_aln_pos + n] = bytes_GAP
            read_aln[cur_aln_pos:cur_aln_pos + n] = read_seq[cur_read_pos:cur_read_pos + n]
            cur_read_pos += n
        elif cigar_el.op == CigarOp.DELETION:
            # consume marker but not query.
            marker_frag_aln[cur_aln_pos:cur_aln_pos + n] = marker.seq.bytes()[cur_marker_pos:cur_marker_pos + n]
            read_aln[cur_aln_pos:cur_aln_pos + n] = bytes_GAP
            cur_marker_pos += n
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
        cur_aln_pos += n

    # ============ Return the appropriate instance.
    return SequenceReadPairwiseAlignment(
        read,
        marker,
        marker_frag_aln,
        read_aln,
        sam_path,
        samline.lineno,
        global_read_start_pos,
        global_read_end_pos,
        global_ref_start_pos,
        global_ref_end_pos,
        samline.is_reverse_complemented
    )


# def reattach_clipped_bases_to_aln(aln: SequenceReadPairwiseAlignment):
#     """
#     For the special case where bases have been clipped but the mapping is not at the edge, append the rest of
#     the fragment.
#     """
#     is_left_edge_mapped = (aln.marker_start == 0)
#     is_right_edge_mapped = (aln.marker_end == len(aln.marker) - 1)
#     n_start_clip = min(aln.soft_clip_start + aln.hard_clip_start, aln.marker_start)
#
#     aligned_marker_seq = aln.marker_frag_aln_with_gaps
#     aligned_read_seq = aln.read_aln_with_gaps
#
#     if aln.reverse_complemented:
#         read_seq = aln.read.seq.revcomp_bytes()
#     else:
#         read_seq = aln.read.seq.bytes()
#
#     if not is_left_edge_mapped and n_start_clip > 0:
#         marker_prefix = aln.marker.seq.bytes()[(aln.marker_start - n_start_clip):aln.marker_start]
#         aligned_marker_seq = np.concatenate([marker_prefix, aligned_marker_seq])
#
#         read_prefix = read_seq[(aln.read_start - n_start_clip):aln.read_start]
#         aligned_read_seq = np.concatenate([read_prefix, aligned_read_seq])
#         if aln.hard_clip_start > 0:
#             aln.hard_clip_start -= n_start_clip
#         elif aln.soft_clip_start > 0:
#             aln.soft_clip_start -= n_start_clip
#         aln.marker_start -= n_start_clip
#         aln.read_start -= n_start_clip
#
#     n_end_clip = min(aln.soft_clip_end + aln.hard_clip_end, len(aln.marker) - aln.marker_end - 1)
#     if not is_right_edge_mapped and n_end_clip > 0:
#         marker_suffix = aln.marker.seq.bytes()[(aln.marker_end + 1):(aln.marker_end + 1 + n_end_clip)]
#         aligned_marker_seq = np.concatenate([aligned_marker_seq, marker_suffix])
#
#         read_suffix = read_seq[(aln.read_end + 1):(aln.read_end + 1 + n_end_clip)]
#         aligned_read_seq = np.concatenate([aligned_read_seq, read_suffix])
#         if aln.hard_clip_end > 0:
#             aln.hard_clip_end -= n_end_clip
#         elif aln.soft_clip_end > 0:
#             aln.soft_clip_end -= n_end_clip
#         aln.marker_end += n_end_clip
#         aln.read_end += n_end_clip
#
#     assert len(aligned_marker_seq.shape) == 1
#     assert len(aligned_read_seq.shape) == 1
#     assert aligned_marker_seq.shape[0] == aligned_read_seq.shape[0]
#     aln.marker_frag_aln_with_gaps = aligned_marker_seq
#     aln.read_aln_with_gaps = aligned_read_seq


def parse_alignments(sam_file: SamIterator,
                     db: StrainDatabase,
                     read_getter: Optional[Callable[[str], SequenceRead]] = None,
                     min_hit_ratio: float = 0.5,
                     min_frag_len: int = 15,
                     print_tqdm_progressbar: bool = True) -> Iterator[SequenceReadPairwiseAlignment]:
    """
    A basic function which parses a SamFile instance and outputs a generator over alignments.
    :param sam_file: The Sam file to parse.
    :param db: The database to create.
    :param read_getter:
    """
    n_alns = sam_file.num_lines()
    logger.debug(f"Parsing {n_alns} alignments from {sam_file.file_path.name}")

    sam_lines = sam_file.mapped_lines()
    if print_tqdm_progressbar:
        from tqdm import tqdm
        sam_lines = tqdm(
            sam_file.mapped_lines(),
            total=n_alns,
            desc=sam_file.file_path.name,
            unit=' sam-line'
        )

    n_mapped_lines = 0
    for samline in sam_lines:
        n_mapped_lines += 1
        try:
            if read_getter is not None:
                # Apply min_hit_ratio criterion.
                if len(samline.read_seq) / len(read_getter(samline.readname)) <= min_hit_ratio:
                    pass

            aln = parse_line_into_alignment(sam_file.file_path,
                                            samline,
                                            db,
                                            read_getter)

            if len(aln.marker_frag) < min_frag_len:
                continue

            yield aln
        except NotImplementedError as e:
            logger.warning(str(e))
    logger.debug(f"{sam_file.file_path.name} -- "
                 f"Total # SAM lines parsed: {n_alns}; "
                 f"# mapped lines: {n_mapped_lines}")
