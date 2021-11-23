from typing import Tuple, List, Iterator
import numpy as np

from chronostrain.model import Marker, Strain, SequenceRead, AbstractMarkerVariant, StrainVariant
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.sequences import *


class FloppMarkerVariant(AbstractMarkerVariant):
    def __init__(self,
                 id: str,
                 base_marker: Marker,
                 seq_with_gaps: SeqType,
                 aln: MarkerMultipleFragmentAlignment,
                 num_supporting_reads: int):
        super().__init__(
            id=id,
            name=base_marker.name,
            seq=seq_with_gaps[seq_with_gaps != nucleotide_GAP_z4],
            base_marker=base_marker,
            metadata=base_marker.metadata
        )
        self.seq_with_gaps = seq_with_gaps
        self.multi_align = aln
        self.num_supporting_reads = num_supporting_reads

    def to_seqrecord(self, description: str = ""):
        return super().to_seqrecord(description=description)

    def get_aligned_reference_region(self, read: SequenceRead, reverse: bool) -> Tuple[SeqType, np.ndarray, np.ndarray, int, int]:
        """
        Returns the aligned fragment (with gaps removed), and a pair of boolean arrays (insertion, deletion).
        The insertion array indicates which positions of the read (with gaps removed) are insertions,
        and the deletion array indicates which positions of the fragment (with gaps removed) are deleted in the read.
        """
        first_idx, last_idx = self.multi_align.aln_gapped_boundary(read, reverse)
        aln = MarkerMultipleFragmentAlignment.delete_double_gaps(
            marker_aln=self.seq_with_gaps[first_idx:last_idx + 1],
            read_aln=self.multi_align.get_aligned_read_seq(read, reverse)[first_idx:last_idx + 1]
        )
        marker_section = aln[0]
        read_section = aln[1]

        insertion_locs = np.equal(marker_section, nucleotide_GAP_z4)
        # Get rid of indices corresponding to deletions.
        insertion_locs = insertion_locs[read_section != nucleotide_GAP_z4]

        deletion_locs = np.equal(read_section, nucleotide_GAP_z4)
        # Get rid of indices corresponding to insertions.
        deletion_locs = deletion_locs[marker_section != nucleotide_GAP_z4]

        start_clip, end_clip = self.multi_align.num_clipped_bases(read, reverse)
        return marker_section[marker_section != nucleotide_GAP_z4], insertion_locs, deletion_locs, start_clip, end_clip

    def subseq_from_read(self, read: SequenceRead) -> Iterator[Tuple[SeqType, np.ndarray, np.ndarray, int, int]]:
        # We already have the alignments from the read to the reference,
        #   so just get the corresponding fragment from this variant.

        if self.multi_align.contains_read(read, False):
            yield self.get_aligned_reference_region(read, False)
        if self.multi_align.contains_read(read, True):
            yield self.get_aligned_reference_region(read, True)

    def subseq_from_pairwise_aln(self, aln):
        raise NotImplementedError("Pairwise alignment to subsequence mapping not implemented, "
                                  "should only be invoked using multiple alignments.")


class FloppStrainVariant(StrainVariant):
    def __init__(self,
                 base_strain: Strain,
                 id: str,
                 variant_markers: List[FloppMarkerVariant],
                 ):
        super().__init__(
            base_strain=base_strain,
            id=id,
            variant_markers=variant_markers
        )
        self.total_num_supporting_reads: int = sum(
            marker.num_supporting_reads
            for marker in variant_markers
        )
