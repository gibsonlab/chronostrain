from typing import Iterable, Tuple, List
import numpy as np

from chronostrain.model import Marker, Strain
from chronostrain.util.alignments import SequenceReadAlignment
from chronostrain.util.sequences import SeqType, map_z4_to_nucleotide


class MarkerVariant(Marker):
    def __init__(self, base_marker: Marker, substitutions: Iterable[Tuple[int, int, int]]):
        """
        :param base_marker: The base marker of which this marker is a variant of.
        :param substitutions: An iterable of (position, base, evidence) tuples.
        """
        self.base_marker = base_marker

        self.quality_evidence = np.sum([qual for _, _, qual in substitutions])

        new_id = "{}<{}>".format(
            base_marker.id,
            "|".join([
                "{}:{}".format(pos, base)
                for pos, base, _ in substitutions
            ])
        )

        new_name: str = "{}-Variant[{}]".format(
            base_marker.name,
            "|".join(["{}:{}".format(pos, map_z4_to_nucleotide(z4base)) for pos, z4base, _ in substitutions])
        )

        new_seq: SeqType = base_marker.seq.copy()
        for pos, z4base, _ in substitutions:
            new_seq[pos] = z4base

        super().__init__(
            id=new_id,
            name=new_name,
            seq=new_seq,
            metadata=base_marker.metadata
        )

    def subseq_from_ref_alignment(self, alignment: SequenceReadAlignment) -> SeqType:
        """
        Retrieve the subsequence of this marker, given the relative positioning (with respect to the base marker)
        stored inside the SequenceReadAlignment instance.

        :param alignment:
        :return: The sequence, returned as a numpy array (SeqType).
        """

        this_marker_start = alignment.marker_start + (ASDFASDF)
        this_marker_end = alignment.marker_end + (ASDF)

        """
        Note:
        The only complexity added is due to indels, otherwise the answer would just be seq[marker_start:marker_end + 1].
        """
        return self.seq[
            this_marker_start:this_marker_end + 1
        ]

    # def subseq_from_base_marker_positions(self, base_marker_start: int, base_marker_end: int) -> SeqType:
    #     return self.seq[base_marker_start:base_marker_end + 1]


class StrainVariant(Strain):
    def __init__(self, marker_variants: List[MarkerVariant], base_strain: Strain):
        self.base_strain = base_strain
        self.quality_evidence = np.sum([marker_variant.quality_evidence for marker_variant in marker_variants])

        new_id = "{}_Variant[{}]".format(
            base_strain.id,
            "+".join([marker_variant.id for marker_variant in marker_variants])
        )

        # Compute the new genome length, assuming that multiple variants of the same marker is coming from increased
        # copy number.
        base_altered_markers = {marker_variant.base_marker for marker_variant in marker_variants}
        base_altered_marker_lengths = np.sum([len(marker.seq) for marker in base_altered_markers])

        variant_marker_lengths = np.sum([len(variant.seq) for variant in marker_variants])
        new_genome_length = base_strain.genome_length - base_altered_marker_lengths + variant_marker_lengths

        base_unaltered_markers = {marker for marker in base_strain.markers}.difference()

        super().__init__(
            id=new_id,
            markers=list(base_unaltered_markers) + marker_variants,
            genome_length=new_genome_length,
            metadata=None
        )

    def __repr__(self):
        return "{}".format(
            super().__repr__()
        )

    def __str__(self):
        return "{}".format(
            super().__str__()
        )
