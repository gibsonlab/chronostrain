from pathlib import Path
from typing import Dict, Tuple, Iterator

import Bio.AlignIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

from chronostrain.model import Marker, SequenceRead
from ...external import mafft_fragment
from ...sequences import *

from chronostrain.config import create_logger
logger = create_logger(__name__)


_READ_PREFIX = "READ"
_SEPARATOR = "#"


class MarkerMultipleAlignment(object):
    """
    Encapsulates a multiple alignment of a marker sequence and short reads.
    """
    def __init__(self,
                 marker: Marker,
                 aligned_marker_seq: SeqType,
                 read_multi_alignment: SeqType,
                 forward_read_index_map: Dict[str, int],
                 reverse_read_index_map: Dict[str, int]
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

    def contains_read_id(self, read_id: str) -> bool:
        return read_id in self.forward_read_index_map or read_id in self.reverse_read_index_map

    def get_alignment(self, read_id: str, reverse: bool) -> SeqType:
        if reverse:
            read_idx = self.reverse_read_index_map[read_id]
        else:
            read_idx = self.forward_read_index_map[read_id]
        return self.delete_double_gaps(self.aligned_marker_seq, self.read_multi_alignment[read_idx, :])

    def get_aligned_reference_region(self, read_id: str, reverse: bool) -> Tuple[SeqType, np.ndarray, np.ndarray]:
        """
        Returns the aligned fragment (with gaps removed), and a pair of boolean arrays (insertion, deletion).
        The insertion array indicates which positions of the read (with gaps removed) are insertions,
        and the deletion array indicates which positions of the fragment (with gaps removed) are deleted in the read.
        """
        aln = self.get_alignment(read_id, reverse)
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

    def read_ids(self, reverse: bool) -> Iterator[str]:
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


def parse(target_marker: Marker, aln_path: Path) -> MarkerMultipleAlignment:
    marker_seq = None
    forward_read_ids = []
    forward_seqs = []

    reverse_read_ids = []
    reverse_seqs = []

    # Parse from file.
    for record_idx, record in enumerate(Bio.AlignIO.read(str(aln_path), 'fasta')):
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
            rev_comp = int(tokens[1]) == 1
            read_id = tokens[2]

            if not rev_comp:
                forward_read_ids.append(read_id)
                forward_seqs.append(nucleotides_to_z4(str(record.seq)))
            else:
                reverse_read_ids.append(read_id)
                reverse_seqs.append(nucleotides_to_z4(str(record.seq)))

    if marker_seq is None:
        raise ValueError(f"Couldn't find marker sequence in specified alignment {str(aln_path)}")

    return MarkerMultipleAlignment(
        marker=target_marker,
        aligned_marker_seq=marker_seq,
        read_multi_alignment=np.stack(forward_seqs + reverse_seqs, axis=0),
        forward_read_index_map={read_id: idx for idx, read_id in enumerate(forward_read_ids)},
        reverse_read_index_map={read_id: idx for idx, read_id in enumerate(reverse_read_ids)}
    )


def align(marker: Marker,
          read_descriptions: Iterator[Tuple[SequenceRead, bool]],
          intermediate_fasta_path: Path,
          out_fasta_path: Path):
    """
    Write these records to file (using a predetermined format), then perform multiple alignment.
    """
    # First write to temporary file, with the reads reverse complemented if necessary.
    records = []
    for read, should_reverse_comp in read_descriptions:
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
            id=f"{_READ_PREFIX}{_SEPARATOR}{revcomp_flag}{_SEPARATOR}{read.id}",
            description=f"(Read, reverse complement:{should_reverse_comp})"
        )
        records.append(record)
    SeqIO.write(records, intermediate_fasta_path, "fasta")

    logger.debug(f"Invoking `mafft --addfragments` on {len(records)} sequences.")

    # Now invoke clustal omega aligner.
    mafft_fragment(
        reference_fasta_path=marker.metadata.file_path,
        fragment_fasta_path=intermediate_fasta_path,
        output_path=out_fasta_path,
        auto=True
    )
