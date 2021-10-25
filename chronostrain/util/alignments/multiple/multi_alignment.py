from pathlib import Path
from typing import Dict, Tuple, Iterator, List

import Bio.AlignIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

from chronostrain.model import Marker, SequenceRead
from chronostrain.model.io import TimeSeriesReads
from ...external import mafft_fragment
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
                 reverse_read_index_map: Dict[SequenceRead, int]
                 ):
        if aligned_marker_seq.shape[0] != read_multi_alignment.shape[1]:
            raise ValueError("Read alignments must be of the same length as the marker alignment string.")
        if read_multi_alignment.shape[0] != len(forward_read_index_map) + len(reverse_read_index_map):
            raise ValueError("Each row of the read multiple alignment must be specified by a read id in either "
                             "the forward or reverse mappings.")

        self.marker = marker
        self.aligned_marker_seq = aligned_marker_seq
        self.read_multi_alignment = read_multi_alignment
        self.forward_read_index_map = forward_read_index_map
        self.reverse_read_index_map = reverse_read_index_map

    def num_bases(self) -> int:
        return len(self.aligned_marker_seq)

    def contains_read_id(self, read_id: str) -> bool:
        return read_id in self.forward_read_index_map or read_id in self.reverse_read_index_map

    def get_alignment(self, read: SequenceRead, reverse: bool, delete_double_gaps: bool = True) -> SeqType:
        if reverse:
            read_idx = self.reverse_read_index_map[read]
        else:
            read_idx = self.forward_read_index_map[read]

        if delete_double_gaps:
            return self.delete_double_gaps(self.aligned_marker_seq, self.read_multi_alignment[read_idx, :])
        else:
            return np.stack([
                self.aligned_marker_seq, self.read_multi_alignment[read_idx, :]
            ], axis=0)

    def get_aligned_reference_region(self, read: SequenceRead, reverse: bool) -> Tuple[SeqType, np.ndarray, np.ndarray]:
        """
        Returns the aligned fragment (with gaps removed), and a pair of boolean arrays (insertion, deletion).
        The insertion array indicates which positions of the read (with gaps removed) are insertions,
        and the deletion array indicates which positions of the fragment (with gaps removed) are deleted in the read.
        """
        aln = self.get_alignment(read, reverse)
        locations = np.where(aln[1] != nucleotide_GAP_z4)[0]
        start = locations[0]
        end = locations[-1]

        align_section = aln[:, start:end+1]
        marker_section = align_section[0]

        insertion_locs = np.equal(align_section[0], nucleotide_GAP_z4)
        insertion_locs = insertion_locs[align_section[1] != nucleotide_GAP_z4]

        deletion_locs = np.equal(align_section[1], nucleotide_GAP_z4)
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

    reverse_reads: List[SequenceRead] = []
    reverse_seqs: List[SeqType] = []

    # Parse from file.
    records = iter(Bio.AlignIO.read(str(aln_path), 'fasta'))

    # First entry should always be the marker.
    first_record = next(records)
    if first_record.id.startswith(f"{_READ_PREFIX}{_SEPARATOR}"):
        raise ValueError(f"Expected Marker aligned sequence, but instead got read identifier {first_record.id}.")

    parsed_marker_id = first_record.id.split("|")[2]
    if target_marker.id != parsed_marker_id:
        raise ValueError(f"Expected marker `{target_marker.id}`, "
                         f"but instead found `{parsed_marker_id}` in alignment file.")
    marker_seq = nucleotides_to_z4(str(first_record.seq))

    # Parse the other entries.
    for record in records:
        record_id: str = record.id

        if not record_id.startswith(f"{_READ_PREFIX}{_SEPARATOR}"):
            # Found marker.
            parsed_marker_id = record_id.split("|")[2]
            if target_marker.id != parsed_marker_id:
                raise ValueError(f"Expected marker `{target_marker.id}`, "
                                 f"but instead found `{parsed_marker_id}` in alignment file.")
            marker_seq = nucleotides_to_z4(str(record.seq))
        else:
            tokens = record_id.split(_SEPARATOR)
            t_idx = int(tokens[1])
            rev_comp = int(tokens[2]) == 1
            read_id = tokens[3]

            read_obj = reads[t_idx].get_read(read_id)

            if not rev_comp:
                forward_reads.append(read_obj)
                forward_seqs.append(nucleotides_to_z4(str(record.seq)))
            else:
                reverse_reads.append(read_obj)
                reverse_seqs.append(nucleotides_to_z4(str(record.seq)))

    # Build the mappings.
    forward_read_index_map = {}
    reverse_read_index_map = {}
    for idx, read in enumerate(forward_reads):
        if read in forward_read_index_map:
            raise RuntimeError(f"Found multiple mapping positions for forward-mapped read {read.id}.")
        forward_read_index_map[read] = idx
    for idx, read in enumerate(reverse_reads):
        if read in reverse_read_index_map:
            raise RuntimeError(f"Found multiple mapping positions for reverse-mapped read {read.id}.")
        reverse_read_index_map[read] = idx

    return MarkerMultipleFragmentAlignment(
        marker=target_marker,
        aligned_marker_seq=marker_seq,
        read_multi_alignment=np.stack(forward_seqs + reverse_seqs, axis=0),
        forward_read_index_map=forward_read_index_map,
        reverse_read_index_map=reverse_read_index_map
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

    logger.debug(f"Invoking `mafft --addfragments` on {len(records)} sequences.")

    # Now invoke MAFFT aligner.
    mafft_fragment(
        reference_fasta_path=marker.metadata.file_path,
        fragment_fasta_path=intermediate_fasta_path,
        output_path=out_fasta_path,
        auto=True
    )
