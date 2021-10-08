from typing import List, Iterator, Tuple, Callable
import numpy as np

from chronostrain.model import Marker
from chronostrain.util.alignments import SequenceReadAlignment
from chronostrain.util.sequences import nucleotides_to_z4, map_z4_to_nucleotide


class TimeSeriesMarkerAlignments(object):
    def __init__(self, num_times: int):
        self.alns = [[] for _ in range(num_times)]

    def add_alignment(self, aln: SequenceReadAlignment, t_idx: int):
        self.alns[t_idx].append(aln)

    def set_alignments(self, alns: List[SequenceReadAlignment], t_idx: int):
        self.alns[t_idx] = alns

    def __iter__(self) -> Iterator[List[SequenceReadAlignment]]:
        yield from self.alns


class MarginalEvidence(object):
    """
    A utility class representing the overall tally (a pileup) of single-nucleotide variants.
    """
    def __init__(self,
                 marker: Marker,
                 alignments: List[SequenceReadAlignment],
                 quality_threshold: float,
                 type_option: str):
        if type_option == "quality":
            self.evidence_matrix = np.zeros(
                shape=(len(marker), 4),
                dtype=float
            )
        elif type_option == "count":
            self.evidence_matrix = np.zeros(
                shape=(len(marker), 4),
                dtype=int
            )
        else:
            raise ValueError("Unknown evidence option `{}`.".format(type_option))
        self.quality_threshold = quality_threshold
        self.add_alignments(alignments)

    def add_alignments(self, alignments: List[SequenceReadAlignment]):
        for aln in alignments:
            read_seq, read_quality = aln.read_aligned_section
            relative_positions_of_variants = np.where(
                (read_seq != aln.marker_frag) & (read_quality > self.quality_threshold)
            )[0]
            bases = read_seq[relative_positions_of_variants]
            self.evidence_matrix[relative_positions_of_variants + aln.marker_start, bases] += 1


class MarkerVariantEvidence(object):
    def __init__(self,
                 marker: Marker,
                 time_series_alignments: TimeSeriesMarkerAlignments,
                 quality_threshold: float):
        self.marker = marker
        self.quality_threshold = quality_threshold
        self.counts_evidence = [
            MarginalEvidence(self.marker, alns_t, self.quality_threshold, "count")
            for alns_t in time_series_alignments
        ]
        self.quality_evidence = [
            MarginalEvidence(self.marker, alns_t, self.quality_threshold, "quality")
            for alns_t in time_series_alignments
        ]

        self.support = self._compute_support(time_series_alignments)

    def _compute_support(self, time_series_alignments: TimeSeriesMarkerAlignments):
        """
        Computes the total evidence across all timepoints, and returns all variants with nonzero # of occurrences.
        :param time_series_alignments:
        :return: A (2 x V) numpy array of dtype int, where each column represents a (position, base) pair.
        """
        total_evidence = MarginalEvidence(self.marker, [], self.quality_threshold, "count")
        for alns_t in time_series_alignments:
            total_evidence.add_alignments(alns_t)
        return np.stack(np.where(total_evidence.evidence_matrix > 0), axis=0)

    def supported_evidence_change(self):
        """
        :return: A (T-1) x (V) numpy matrix of timepoint differences in evidences. It is assumed (for
            downstream calculations) that if two variants are from the same marker, then their count
            changes are correlated.
        """
        return np.diff(
            np.stack([
                evidence_t.evidence_matrix[self.support[0, :], self.support[1, :]]
                for evidence_t in self.counts_evidence
            ], axis=0),
            axis=0
        )

    def num_supported_variants(self) -> int:
        """
        :return: The number of supported variants.
        """
        return self.support.shape[1]

    def supported_variants(self) -> Iterator[Tuple[int, int]]:
        """
        :return: A generator over the supported variants, represented as a (position, z4-base) tuple.
        """
        for i in range(self.num_supported_variants()):
            yield self.support[0, i], self.support[1, i]

    def variant_desc(self) -> Iterator[str]:
        """
        :return: A generator yielding a informative representation of each tuple. For debugging purposes.
        """
        for pos, base in self.supported_variants():
            yield "{pos}:{ref}->{var}".format(
                pos=pos,
                ref=self.marker.nucleotide_seq[pos],
                var=map_z4_to_nucleotide(base)
            )

    @property
    def variant_getter(self) -> Callable[[int], Tuple[int, int]]:
        def my_getter(variant_idx: int):
            return self.support[0, variant_idx], self.support[1, variant_idx]
        return my_getter
