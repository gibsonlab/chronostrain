import itertools
from collections import defaultdict
from typing import List, Iterator, Tuple, Dict, Callable
import numpy as np
from scipy.linalg import eigh
import networkx as nx

from chronostrain.algs.subroutines import CachedReadPairwiseAlignments
from chronostrain.database import StrainDatabase
from chronostrain.model import Marker
from chronostrain.model.io import TimeSeriesReads

from .evidence import MarkerVariantEvidence, TimeSeriesMarkerAlignments, MarginalVariantQualityEvidence
from .base import MarkerVariant, StrainVariant

from chronostrain.config import create_logger
logger = create_logger(__name__)


def entrywise_sum(*arrays):
    ans = np.zeros(shape=arrays[0].shape, dtype=arrays[0].dtype)
    for arr in arrays:
        ans = ans + arr
    return ans


def upper_triangular_upper_bounded(x: np.ndarray,
                                   upper_bound: float,
                                   k: int = 0
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the indices satisfying the condition x < upper_bound only on the upper triangular part of
    the provided array (with diagonal offset k).

    Is equivalent to performing np.where(x > upper_bound), then filtering according to whether row + k <= col.
    :return: A tuple of numpy arrays, representing the array of valid rows and the matching array valid columns.
    """
    r, c = np.triu_indices_from(x, k=k)
    values = x[r, c]
    valid_indices = np.where(values < upper_bound)
    return r[valid_indices], c[valid_indices]


def pairwise_euclidean_distance(x: np.ndarray):
    """
    Given an (N x D) matrix, interpret each row of input matrix as a point in R^D.
    Compute the (N x N) euclidean distance matrix.

    For efficiency, the computation performed is a vectorized adaptation of the formula d(x,y) = |x| + |y| - 2<x,y>.

    :param x: A stack of row vectors, where the i-th row are the coordinates of the i-th point.
    :return: An (N x N) matrix.
    """
    inner_prods = x @ x.transpose()
    norms = np.diag(inner_prods)
    diff = norms.reshape(-1, 1) - inner_prods
    return diff + diff.transpose()


class NoVariantsException(BaseException):
    pass


class StrainVariantComputer(object):
    def __init__(self,
                 db: StrainDatabase,
                 reads: TimeSeriesReads,
                 quality_threshold: float,
                 eig_lower_bound: float = 1e-6,
                 variant_distance_upper_bound: float = 1e-8):
        self.db = db
        self.all_markers = db.all_markers()
        self.reads = reads
        self.quality_threshold = quality_threshold
        self.eig_lower_bound = eig_lower_bound
        self.variant_distance_upper_bound = variant_distance_upper_bound
        self.cached_alignments = CachedReadPairwiseAlignments(reads=self.reads, db=self.db)

    def construct_variants(self) -> Iterator[StrainVariant]:
        """
        :return: An iterator over variants (a 1-d numpy array containing a marker sequence.
        """
        G, variant_pair_getter, quality_evidences = self._construct_similarity_graph()
        for clique in nx.find_cliques(G):
            yield self._strain_variant_from_clique(clique, variant_pair_getter, quality_evidences)

    def _strain_variant_from_clique(
            self,
            clique: List[int],
            variant_pair_getter: Callable[[int], Tuple[int, int, int, int, int]],
            quality_evidences: List[List[MarginalVariantQualityEvidence]]
    ) -> StrainVariant:
        """
        :param clique: The list of variant indices representing a clique.
        :param variant_pair_getter: A callable instance representing the semantic mapping
            (cumulative 2-NV index) -> (Marker, pos1, base1, pos2, base2).
        :param quality_evidences: A marker-indexed list of time-series marginal quality score evidences.
        :return: A generator yielding resolutions of the clique into variants.
        """
        # For each marker, tally up the total evidence across time.
        total_marker_evidences = [
            entrywise_sum(*(marker_evidence_t.matrix for marker_evidence_t in marker_evidences))
            for marker_evidences in quality_evidences
        ]

        # Group together the variants in the clique by their marker.
        markers_to_subcliques = defaultdict(list)
        for paired_variant_idx in clique:
            target_marker_idx, pos1, base1, pos2, base2 = variant_pair_getter(paired_variant_idx)
            markers_to_subcliques[target_marker_idx].append((pos1, base1))
            markers_to_subcliques[target_marker_idx].append((pos2, base2))

        # Instantiate the marker variants.
        # By concatenating all the possible combinatorial constructions from the clique into the same strain, the
        # interpretation here is that multiple marker variants with very high correlation are due to copy numbers.
        marker_variants: List[MarkerVariant] = []
        for marker_idx, subclique in markers_to_subcliques.items():
            marker = self.all_markers[marker_idx]
            marker_variants += list(
                self._marker_variants_from_clique(marker, subclique, total_marker_evidences[marker_idx])
            )

        # Determine the base strain using the best-matching marker.
        base_strain = self.db.best_matching_strain([
            marker_variant.base_marker for marker_variant in marker_variants
        ])

        # Return the new strain instance.
        return StrainVariant(marker_variants=marker_variants, base_strain=base_strain)

    def _marker_variants_from_clique(
            self,
            base_marker: Marker,
            clique: List[Tuple[int, int]],
            quality_evidences: np.ndarray
    ) -> Iterator[MarkerVariant]:
        """
        :param base_marker: The base marker to build from.
        :param clique: The clique of mutually correlated variants, represented by (position, base) pairs.
        :param quality_evidences: A (L x 4) array of totaled quality evidence scores.
        :return: A MarkerVariant instance.
        """
        # For each position, tally up which variants it has, if any.
        variants = defaultdict(list)
        for pos, base in clique:
            variants[pos].append(base)

        def nucleotide_iterator(pos, base_list):
            for base in base_list:
                yield pos, base, quality_evidences[pos, base]

        # Take all possible combinatorial combinations.
        for variant_desc in itertools.product(
                *(nucleotide_iterator(pos, base_list)
                  for pos, base_list in variants.items())
        ):
            # Ignore reference (position, base) pairs.
            substitutions = [
                substitution
                for substitution in variant_desc
                if base_marker.seq[substitution[0]] != substitution[1]
            ]

            # At this point, the positions are guaranteed to be unique.
            yield MarkerVariant(
                base_marker=base_marker,
                substitutions=substitutions
            )

    def _construct_similarity_graph(
            self
    ) -> Tuple[
        nx.Graph,
        Callable[[int], Tuple[int, int, int, int, int]],
        List[List[MarginalVariantQualityEvidence]]
    ]:
        """
        Computes the similarity graph based on the inner-product similarity measure on the embedding space.
        The embedding is the nonzero eigenvector set of the correlation matrix.

        :return: A tuple of three outputs:
        1) The Graph instance, where edges imply high correlation between pairs of variants.
        2)
        """
        # ==== Retrieve alignments and initialize necessary values.
        all_obs = []

        # ==== To be used in identifying variant indices across different markers.
        sizes = np.zeros(shape=len(self.all_markers), dtype=int)
        variant_getters = []
        quality_evidences = []

        # ==== Normalization for computing frequencies from counts.
        read_depth_normalization = np.array([
            1 / self.reads.time_slices[t_idx].read_depth
            for t_idx in range(len(self.reads))
        ])

        # ==== Tally up evidence for variants.
        for marker_idx, (marker, alns) in enumerate(self._alignments_by_marker(self.all_markers)):
            # Create the observation matrix for this marker, to be concatenated.
            variants = MarkerVariantEvidence(marker, alns, self.quality_threshold)
            obs = variants.timeseries_pairwise_counts_matrix() * read_depth_normalization
            all_obs.append(obs)
            logger.debug("Marker {}: # variant pairs: {}".format(marker.name, len(variants.relevant_pairs)))

            # To be used in making the variant idx -> variant mapping.
            assert obs.shape[0] == len(self.reads)
            assert obs.shape[1] == len(variants.relevant_pairs)
            sizes[marker_idx] = len(variants.relevant_pairs)
            variant_getters.append(variants.variant_pair_getter)

            # To be used in determining the evidence of variants.
            quality_evidences.append(variants.marginal_evidences)

        cumulative_sizes = np.cumsum(sizes)  # To be used in identifying variant indices across different markers.

        # ==== A callable which maps the cumulative variant index into the individual marker's relative variant index.
        def variant_pair_getter(cumulative_variant_idx: int) -> Tuple[int, int, int, int, int]:
            # Try to figure out the marker index based on the cumulative variant index.
            try:
                tgt_marker_idx = np.where(cumulative_variant_idx < cumulative_sizes)[0][0]
            except IndexError as e:
                raise IndexError(
                    "Cannot index cumulative marker index {}, which exceeds the total number of variants {}.".format(
                        cumulative_variant_idx, cumulative_sizes[-1]
                    )
                ) from e

            # Convert the cumulative variant index into an index relative to the marker.
            if tgt_marker_idx > 0:
                relative_idx = cumulative_variant_idx - cumulative_sizes[tgt_marker_idx - 1]
            else:
                relative_idx = cumulative_variant_idx

            # Invoke the getter.
            pos1, base1, pos2, base2 = variant_getters[tgt_marker_idx](relative_idx)
            return tgt_marker_idx, pos1, base1, pos2, base2

        # ==== The (T x V^2) master frequency matrix.
        freqs = np.concatenate(all_obs, axis=1)
        logger.debug("Frequency matrix dim: {}".format(freqs.shape))

        if freqs.shape[1] == 0:
            raise NoVariantsException("No inferrable variants from provided quality threshold setting {}.".format(
                self.quality_threshold
            ))

        # ==== Only report eigenvalues > eig_lower_bound.
        evals, evecs = eigh(freqs.transpose() @ freqs, subset_by_value=[self.eig_lower_bound, np.inf])
        if len(evals) == 0:
            raise NoVariantsException(
                "No inferrable variants from eigenspace decomposition. "
                "Try lowering eigenvalue tolerance from {}.".format(
                    self.eig_lower_bound
                )
            )
        logger.debug("# evecs passing evalue threshold {:.2e}: {}".format(self.eig_lower_bound, len(evals)))

        # Compute Euclidean distance between embedded points.
        pairwise_distances = pairwise_euclidean_distance(evecs)

        # Use all entries above main diagonal to determine similarity.
        similar_variant_pairs = zip(
            *upper_triangular_upper_bounded(
                x=pairwise_distances,
                upper_bound=self.variant_distance_upper_bound,
                k=1
            )
        )

        G = nx.Graph()
        for v in range(pairwise_distances.shape[0]):
            G.add_node(v)
        G.add_edges_from(similar_variant_pairs)

        logger.debug("# of similar pairs (threshold < {:.2e}): {}".format(
            self.variant_distance_upper_bound,
            len(G.edges())
        ))

        if len(G.edges) == 0:
            raise NoVariantsException("No inferrable variants from embedding-space similarity threshold {}.".format(
                self.variant_distance_upper_bound
            ))
        return G, variant_pair_getter, quality_evidences

    def _alignments_by_marker(self, markers: List[Marker]) -> Iterator[Tuple[Marker, TimeSeriesMarkerAlignments]]:
        """
        Restructure the List-of-lists representation of alignments by grouping them by markers for each t.
        """
        markers_to_alns: Dict[Marker, TimeSeriesMarkerAlignments] = {
            marker: TimeSeriesMarkerAlignments(num_times=len(self.reads))
            for marker in markers
        }

        for t_idx in range(len(self.reads)):
            alns_t = self.cached_alignments.alignments_by_marker_and_timepoint(t_idx)
            read_depth_t = self.reads.time_slices[t_idx].read_depth
            for marker, alns in alns_t.items():
                markers_to_alns[marker].set_alignments(alns, read_depth_t, t_idx)

        yield from markers_to_alns.items()
