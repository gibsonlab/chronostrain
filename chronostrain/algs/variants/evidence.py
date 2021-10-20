from collections import defaultdict
from typing import List, Iterator, Tuple, Callable, Iterable, Set
import numpy as np
import itertools

from chronostrain.model import Marker
from chronostrain.util.alignments import SequenceReadAlignment
from chronostrain.util.sequences import nucleotide_N_z4


class TimeSeriesMarkerAlignments(object):
    def __init__(self, num_times: int):
        self.alns: List[List[SequenceReadAlignment]] = [[] for _ in range(num_times)]
        self.read_depths: List[int] = [0 for _ in range(num_times)]

    def set_alignments(self, alns: List[SequenceReadAlignment], read_depth: int, t_idx: int):
        self.alns[t_idx] = alns
        self.read_depths[t_idx] = read_depth

    def __iter__(self) -> Iterator[Tuple[List[SequenceReadAlignment], int]]:
        yield from zip(self.alns, self.read_depths)


class MarginalVariantQualityEvidence(object):
    def __init__(self,
                 marker: Marker,
                 alignments: Iterable[SequenceReadAlignment],
                 quality_threshold: float):
        """
        Stores the total quality score of each observed variant (e.g. the read's base does not match the reference
        base).

        :param marker:
        :param alignments:
        :param quality_threshold:
        """
        self.quality_threshold = quality_threshold
        self.matrix = self.create_evidence_matrix(marker, alignments)

    def create_evidence_matrix(self, marker: Marker, alignments: Iterable[SequenceReadAlignment]) -> np.ndarray:
        m = np.zeros(shape=(len(marker), 4), dtype=float)

        # ============== Parsing ==============
        for aln in alignments:
            read_seq, read_quality = aln.read_aligned_section(delete_indels=ASLKDF)
            marker_frag = aln.marker_aligned_frag(delete_indels=ASDFASD)
            assert len(read_seq) == len(marker_frag)

            relative_variant_positions = np.where(
                (read_seq != nucleotide_N_z4) & (read_seq != marker_frag) & (
                            read_quality > self.quality_threshold)
            )[0]
            variant_bases = read_seq[relative_variant_positions]
            variant_quals = read_quality[relative_variant_positions]
            m[relative_variant_positions + aln.marker_start, variant_bases] += variant_quals

        return m

    @property
    def supported_positions(self) -> Set[int]:
        return set(np.where(
            np.sum(self.matrix, axis=1) > 0
        )[0])


class PairwiseFrequencyEvidence(object):
    """
    A utility class representing the overall tally (a pileup) of single-nucleotide variants.
    """
    def __init__(self,
                 supported_positions: Set[int],
                 alignments: Iterable[SequenceReadAlignment],
                 quality_threshold: float):
        self.quality_threshold = quality_threshold
        self.supported_positions = supported_positions
        self.pairwise_counts = defaultdict(int)
        self._count_pairwise_occurrences(alignments)

    @property
    def pairwise_support(self):
        return self.pairwise_counts.keys()

    @staticmethod
    def lexicographic_ordering_bases(pos: int, z4_base: int):
        return (4 * pos) + z4_base

    def _get_count(self, pos1: int, base1: int, pos2: int, base2: int) -> int:
        # Only store nondecreasing pairs.
        if self.lexicographic_ordering_bases(pos1, base1) > self.lexicographic_ordering_bases(pos2, base2):
            return self._get_count(pos2, base2, pos1, base1)
        else:
            return self.pairwise_counts[(pos1, base1, pos2, base2)]

    def _increment(self, pos1, base1, pos2, base2):
        # Only store nondecreasing pairs.
        if self.lexicographic_ordering_bases(pos1, base1) > self.lexicographic_ordering_bases(pos2, base2):
            self._increment(pos2, base2, pos1, base1)
        else:
            self.pairwise_counts[(pos1, base1, pos2, base2)] += 1

    def specified_counts(self, variant_pairs: List[Tuple[int, int, int, int]]) -> np.ndarray:
        return np.array([
            self._get_count(*pair) for pair in variant_pairs
        ])

    def _count_pairwise_occurrences(self, alignments: Iterable[SequenceReadAlignment]):
        # ============= Helper functions =============
        def supported_positions(positions: Iterable[int]) -> Iterator[int]:
            for position in positions:
                if position in self.supported_positions:
                    yield position

        # ============== Parsing ==============
        """
        Add to the matrix all observed high-quality pairs of nucleotides at the supported positions.
        Includes reference nucleotides, since the lack of a variant observed at position p is also a signal 
        (indicating negative correlation).
        """
        for aln in alignments:
            read_seq, read_quality = aln.read_aligned_section(delete_indels=ASDFASDF)
            high_quality_positions = np.where(
                (read_seq != nucleotide_N_z4) & (read_quality > self.quality_threshold)
            )[0]
            marker_frag = aln.marker_aligned_frag(delete_indels=True)
            # TODO indels

            for pos1, pos2 in itertools.combinations(supported_positions(high_quality_positions), r=2):
                base1 = read_seq[pos1]
                base2 = read_seq[pos2]

                ref_base1 = marker_frag[pos1]
                ref_base2 = marker_frag[pos2]
                if (ref_base1 == base1) and (ref_base2 == base2):
                    # Both positions are reference bases; to save space, ignore these.
                    continue

                self._increment(pos1, base1, pos2, base2)


class MarkerVariantEvidence(object):
    def __init__(self,
                 marker: Marker,
                 time_series_alignments: TimeSeriesMarkerAlignments,
                 quality_threshold: float):
        """
        Encapsulates the other two Evidence classes.

        :param marker:
        :param time_series_alignments:
        :param quality_threshold:
        """
        # TODO: have MarginalEvidence record deletions and insertions, using the alignments.
        self.marker = marker
        self.quality_threshold = quality_threshold

        self.marginal_evidences = [
            MarginalVariantQualityEvidence(self.marker, alns_t, self.quality_threshold)
            for alns_t, read_depth_t in time_series_alignments
        ]

        self.supported_positions = set()
        for ev in self.marginal_evidences:
            self.supported_positions = self.supported_positions.union(ev.supported_positions)

        self.pairwise_evidences = [
            PairwiseFrequencyEvidence(self.supported_positions, alns_t, self.quality_threshold)
            for alns_t, read_depth_t in time_series_alignments
        ]

        relevant_pairs_set = set()
        for ev in self.pairwise_evidences:
            relevant_pairs_set = relevant_pairs_set.union(ev.pairwise_support)
        self.relevant_pairs: List[Tuple[int, int, int, int]] = list(relevant_pairs_set)

    def timeseries_pairwise_counts_matrix(self):
        """
        :return: A (T) x (V^2) numpy matrix of timepoint-wise counts of each relevant pair.
        """
        return np.stack([
            evidence_t.specified_counts(self.relevant_pairs)
            for evidence_t in self.pairwise_evidences
        ], axis=0)

    @property
    def variant_pair_getter(self) -> Callable[[int], Tuple[int, int, int, int]]:
        def my_getter(variant_pair_idx: int):
            return self.relevant_pairs[variant_pair_idx]
        return my_getter
