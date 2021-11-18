from pathlib import Path
from typing import Dict, Tuple, Iterator, List

import Bio.AlignIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

from chronostrain.model import Marker, SequenceRead
from chronostrain.model.io import TimeSeriesReads
from ...external import clustal_omega
from ...sequences import *

from chronostrain.config import create_logger
logger = create_logger(__name__)


_READ_PREFIX = "READ"
_SEPARATOR = "#"


class MarkerMultipleFragmentAlignment(object):
    """
    Encapsulates a multiple alignment of a marker sequence and short reads.
    """
    def __init__(self,
                 marker: Marker,
                 aligned_marker_seq: SeqType,
                 read_multi_alignment: SeqType,
                 forward_read_index_map: Dict[SequenceRead, int],
                 reverse_read_index_map: Dict[SequenceRead, int],
                 time_idxs: np.ndarray
                 ):
        if aligned_marker_seq.shape[0] != read_multi_alignment.shape[1]:
            raise ValueError("Read alignments must be of the same length as the marker alignment string.")
        if read_multi_alignment.shape[0] != len(forward_read_index_map) + len(reverse_read_index_map):
            raise ValueError("Each row of the read multiple alignment must be specified by a read id in either "
                             "the forward or reverse mappings.")

        self.marker = marker
        self.aligned_marker_seq = aligned_marker_seq
        self.read_multi_alignment = read_multi_alignment
        self.time_idxs = time_idxs
        self.forward_read_index_map = forward_read_index_map
        self.reverse_read_index_map = reverse_read_index_map

    def num_bases(self) -> int:
        return len(self.aligned_marker_seq)

    def contains_read(self, read: SequenceRead, reverse: bool) -> bool:
        if reverse:
            return read in self.reverse_read_index_map
        else:
            return read in self.forward_read_index_map

    def get_index_of(self, read: SequenceRead, reverse: bool) -> int:
        if reverse:
            return self.reverse_read_index_map[read]
        else:
            return self.forward_read_index_map[read]

    def get_aligned_read_seq(self, read: SequenceRead, reverse: bool) -> SeqType:
        read_idx = self.get_index_of(read, reverse)
        return self.read_multi_alignment[read_idx, :]

    def get_alignment(self, read: SequenceRead, reverse: bool, delete_double_gaps: bool = True) -> SeqType:
        read_seq = self.get_aligned_read_seq(read, reverse)

        if delete_double_gaps:
            return self.delete_double_gaps(self.aligned_marker_seq, read_seq)
        else:
            return np.stack([
                self.aligned_marker_seq, read_seq
            ], axis=0)

    def aln_gapped_boundary(self, read: SequenceRead, revcomp: bool) -> Tuple[int, int]:
        """
        Find the first and last ungapped positions.
        :return: A tuple (i, j) such that aln_seq[i] != GAP and aln_seq[j] != GAP, and all elements to the left of i, and
        to the right of j, are GAPs.
        """
        if revcomp:
            r_idx = self.reverse_read_index_map[read]
        else:
            r_idx = self.forward_read_index_map[read]
        return self.aln_gapped_boundary_of_row(r_idx)

    def aln_gapped_boundary_of_row(self, row_idx: int) -> Tuple[int, int]:
        aln_seq = self.read_multi_alignment[row_idx]
        ungapped_indices = np.where(aln_seq != nucleotide_GAP_z4)[0]
        return ungapped_indices[0], ungapped_indices[-1]

    def get_aligned_reference_region(self, read: SequenceRead, reverse: bool) -> Tuple[SeqType, np.ndarray, np.ndarray]:
        """
        Returns the aligned fragment (with gaps removed), and a pair of boolean arrays (insertion, deletion).
        The insertion array indicates which positions of the read (with gaps removed) are insertions,
        and the deletion array indicates which positions of the fragment (with gaps removed) are deleted in the read.
        """
        aln = self.get_alignment(read, reverse, delete_double_gaps=True)
        first, last = self.aln_gapped_boundary(read, reverse)

        align_section = aln[:, first:last+1]
        marker_section = align_section[0]

        insertion_locs = np.equal(align_section[0], nucleotide_GAP_z4)
        # Get rid of indices corresponding to deletions.
        insertion_locs = insertion_locs[align_section[1] != nucleotide_GAP_z4]

        deletion_locs = np.equal(align_section[1], nucleotide_GAP_z4)
        # Get rid of indices corresponding to insertions.
        deletion_locs = deletion_locs[align_section[0] != nucleotide_GAP_z4]

        return marker_section[marker_section != nucleotide_GAP_z4], insertion_locs, deletion_locs

    def reads(self, reverse: bool) -> Iterator[SequenceRead]:
        if reverse:
            yield from self.reverse_read_index_map.keys()
        else:
            yield from self.forward_read_index_map.keys()

    @staticmethod
    def delete_double_gaps(marker_aln: SeqType, read_aln: SeqType) -> SeqType:
        """
        Eliminate from the pair of alignment strings the indices where both sequences simultaneously have gaps.
        """
        ungapped_indices = (marker_aln != nucleotide_GAP_z4) | (read_aln != nucleotide_GAP_z4)
        return np.stack([
            marker_aln[ungapped_indices], read_aln[ungapped_indices]
        ], axis=0)


def parse(target_marker: Marker, reads: TimeSeriesReads, aln_path: Path) -> MarkerMultipleFragmentAlignment:
    forward_reads: List[SequenceRead] = []
    forward_seqs: List[SeqType] = []
    forward_time_idxs: List[int] = []

    reverse_reads: List[SequenceRead] = []
    reverse_seqs: List[SeqType] = []
    reverse_time_idxs: List[int] = []

    # ============================ BEGIN HELPERS ============================
    def parse_marker_record(marker_record: SeqRecord) -> Tuple[int, int, SeqType]:
        parsed_marker_id = marker_record.id.split("|")[2]
        if target_marker.id != parsed_marker_id:
            raise ValueError(f"Expected marker `{target_marker.id}`, "
                             f"but instead found `{parsed_marker_id}` in alignment file.")
        aligned_marker_seq = nucleotides_to_z4(str(marker_record.seq))

        # The marker sequence's aligned region. Keep track of this to clip off the start/end edge effects.
        matched_indices = np.where(aligned_marker_seq != nucleotide_GAP_z4)[0]
        start_clip = matched_indices[0]
        end_clip = matched_indices[-1]
        marker_seq = aligned_marker_seq[start_clip:end_clip + 1]

        # The marker sequence's aligned region. Keep track of this to clip off the start/end edge effects.
        matched_indices = np.where(marker_seq != nucleotide_GAP_z4)[0]
        marker_start = matched_indices[0]
        marker_end = matched_indices[-1]
        aligned_marker_seq = aligned_marker_seq[start_clip:end_clip + 1]
        return marker_start, marker_end, aligned_marker_seq

    def parse_read_record(read_record: SeqRecord, start_clip: int, end_clip: int):
        # Parse the tokens in the ID.
        tokens = read_record.id.split(_SEPARATOR)
        t_idx = int(tokens[1])
        rev_comp = int(tokens[2]) == 1
        read_id = tokens[3]

        # Get the read instance.
        read_obj = reads[t_idx].get_read(read_id)

        # Check if alignment is clipped off the edges of the marker.
        aln_seq = nucleotides_to_z4(str(read_record.seq))
        n_clipped_bases = np.sum(
            aln_seq[:start_clip] != nucleotide_GAP_z4
        ) + np.sum(
            aln_seq[end_clip + 1:] != nucleotide_GAP_z4
        )
        if n_clipped_bases > 0:
            logger.debug(f"Skipping alignment of read {read_id} in multiple alignment, "
                         f"due to {n_clipped_bases} clipped bases.")
            return

        aln_seq = aln_seq[start_clip:end_clip + 1]

        # Store the alignment into the proper category (whether or not the read was reverse complemented).
        if not rev_comp:
            forward_reads.append(read_obj)
            forward_seqs.append(aln_seq)
            forward_time_idxs.append(t_idx)
        else:
            reverse_reads.append(read_obj)
            reverse_seqs.append(aln_seq)
            reverse_time_idxs.append(t_idx)
    # ============================ END HELPERS ============================
    # Parse the marker entry first.
    start_clip = None
    end_clip = None
    marker_seq = []
    for record in Bio.AlignIO.read(str(aln_path), 'fasta'):
        if not record.id.startswith(f"{_READ_PREFIX}{_SEPARATOR}"):
            start_clip, end_clip, marker_seq = parse_marker_record(record)

    if start_clip is None or end_clip is None:
        raise RuntimeError("Couldn't find reference marker in multiple alignment output {}".format(
            aln_path
        ))

    # Parse the other entries.
    for record in Bio.AlignIO.read(str(aln_path), 'fasta'):
        if record.id.startswith(f"{_READ_PREFIX}{_SEPARATOR}"):
            parse_read_record(record, start_clip, end_clip)

    # Build the mappings.
    forward_read_index_map = {}
    reverse_read_index_map = {}
    for idx, read in enumerate(forward_reads):
        if read in forward_read_index_map:
            raise RuntimeError(f"Found repeat entry forward-mapped read {read.id}.")
        forward_read_index_map[read] = idx
    for idx, read in enumerate(reverse_reads):
        if read in reverse_read_index_map:
            raise RuntimeError(f"Found repeat entry for reverse-mapped read {read.id}.")
        reverse_read_index_map[read] = idx + len(forward_reads)

    return MarkerMultipleFragmentAlignment(
        marker=target_marker,
        aligned_marker_seq=marker_seq,
        read_multi_alignment=np.stack(forward_seqs + reverse_seqs, axis=0),
        forward_read_index_map=forward_read_index_map,
        reverse_read_index_map=reverse_read_index_map,
        time_idxs=np.array(forward_time_idxs + reverse_time_idxs, dtype=int),
    )


def align(marker: Marker,
          read_descriptions: Iterator[Tuple[int, SequenceRead, bool]],
          intermediate_fasta_path: Path,
          out_fasta_path: Path):
    """
    Write these records to file (using a predetermined format), then perform multiple alignment.
    """
    # First write to temporary file, with the reads reverse complemented if necessary.
    records = []
    for t_idx, read, should_reverse_comp in read_descriptions:
        if should_reverse_comp:
            read_seq = reverse_complement_seq(read.seq)
        else:
            read_seq = read.seq

        if should_reverse_comp:
            revcomp_flag = 1
        else:
            revcomp_flag = 0

        record = SeqRecord(
            Seq(z4_to_nucleotides(read_seq)),
            id=f"{_READ_PREFIX}{_SEPARATOR}{t_idx}{_SEPARATOR}{revcomp_flag}{_SEPARATOR}{read.id}",
            description=f"(Read, t: {t_idx}, reverse complement:{should_reverse_comp})"
        )
        records.append(record)
    SeqIO.write(records, intermediate_fasta_path, "fasta")

    logger.debug(
        f"Invoking `clustalo` on {len(records)} sequences, using {marker.metadata.file_path.name} as profile."
    )

    # Now invoke Clustal-Omega aligner.
    clustal_omega(
        input_path=intermediate_fasta_path,
        output_path=out_fasta_path,
        force_overwrite=True,
        verbose=False,
        profile1=marker.metadata.file_path,
        out_format='fasta',
        seqtype='DNA',
        guidetree_out=out_fasta_path.parent / "guidetree"
    )
