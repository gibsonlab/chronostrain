"""
    glopp_with_corr.py
    An algorithm which resolves strain variants (metagenomic "haplotypes") using a combination of glopp and time-series
    correlation grouping.
"""
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, Iterator, Union, Iterable, Optional

import numpy as np

from sklearn import covariance, preprocessing, linear_model
from sklearn.decomposition import NMF
import networkx as nx

from chronostrain.config import create_logger
from chronostrain.database import StrainDatabase
from chronostrain.algs.subroutines.assembly import CachedGloppVariantAssembly, FloppMarkerAssembly, FloppMarkerContig
from chronostrain.model import AbstractMarkerVariant, StrainVariant
from .base import FloppMarkerVariant, FloppStrainVariant
from ...model import Marker
from ...model.io import TimeSeriesReads
from .meta import AbstractVariantBBVISolver

logger = create_logger(__name__)


class GloppContigStrandSpecification(object):
    def __init__(self, contig: FloppMarkerContig, strand_idx: int):
        self.contig = contig
        self.strand_idx = strand_idx

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "<" \
               f"Marker {self.contig.canonical_marker.id}, " \
               f"Contig {self.contig.contig_idx}, " \
               f"Strand {self.strand_idx}" \
               ">"


def upper_triangular_bounded(x: np.ndarray,
                             lower_bound: float,
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
    valid_indices = np.where((values < upper_bound) & (values > lower_bound))
    return r[valid_indices], c[valid_indices]


def partial_corr_matrix(precision: np.ndarray):
    """
    https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion

    :param precision:
    :return:
    """
    diag = np.sqrt(np.diag(precision)).reshape(-1, 1)
    return -precision / diag / diag.transpose()


class GloppVariantSolver(AbstractVariantBBVISolver):
    def __init__(self,
                 db: StrainDatabase,
                 reads: TimeSeriesReads,
                 time_points: List[float],
                 bbvi_iters: int,
                 bbvi_lr: int,
                 bbvi_num_samples: int,
                 quality_lower_bound: float,
                 variant_count_lower_bound: int,
                 glasso_shrinkage: float = 0.1,
                 glasso_standardize: bool = True,
                 glasso_alpha: float = 1e-3,
                 glasso_iterations: int = 5000,
                 glasso_tol: float = 1e-3,
                 seed_with_database: bool = False,
                 num_strands: Optional[int] = None):
        """
        :param glasso_standardize: Indicates whether to standardize each feature across samples.
        :param glasso_alpha: The `alpha` regularization parameter in the glasso algorithm.
        """
        super().__init__(
            db=db,
            reads=reads,
            time_points=time_points,
            bbvi_iters=bbvi_iters,
            bbvi_lr=bbvi_lr,
            bbvi_num_samples=bbvi_num_samples
        )
        self.quality_lower_bound = quality_lower_bound
        self.variant_count_lower_bound = variant_count_lower_bound
        self.reference_markers = self.db.all_canonical_markers()
        self.glasso_shrinkage = glasso_shrinkage
        self.glasso_standardize = glasso_standardize
        self.glasso_alpha = glasso_alpha
        self.glasso_iterations = glasso_iterations
        self.glasso_tol = glasso_tol
        self.num_strands = num_strands
        # self.partial_corr_lower_bound = -0.1
        self.nmf_lower_bound = 0.5

        self.reference_markers_to_assembly: Dict[Marker, FloppMarkerAssembly] = self.construct_marker_assemblies()

        """
        Obtain the mapping (absolute strand idx) -> (target strand)
        """
        self.strand_specs = [
            GloppContigStrandSpecification(contig, strand_idx)
            for assembly in self.assemblies
            for contig in assembly.contigs
            for strand_idx in range(contig.num_strands)
        ]

    def construct_marker_assemblies(self) -> Dict[Marker, FloppMarkerAssembly]:
        marker_assemblies = {}
        for marker_multi_align in self.multi_alignments:
            marker_assembly: FloppMarkerAssembly = CachedGloppVariantAssembly(
                self.reads,
                marker_multi_align,
                quality_lower_bound=self.quality_lower_bound,
                variant_count_lower_bound=self.variant_count_lower_bound
            ).run(num_variants=self.num_strands)

            marker_assemblies[marker_assembly.canonical_marker] = marker_assembly
        return marker_assemblies

    @property
    def assemblies(self) -> Iterator[FloppMarkerAssembly]:
        for marker in self.reference_markers:
            yield self.reference_markers_to_assembly[marker]

    def propose_variants(self, used_variants: List[StrainVariant]) -> Iterator[StrainVariant]:
        variants: List[FloppStrainVariant] = list(
            self.construct_variants_using_assembly(
                # partial_corr_lower_bound=self.partial_corr_lower_bound
                nmf_lower_bound=self.nmf_lower_bound
            )
        )

        def sort_key(v: FloppStrainVariant) -> int:
            return v.total_num_supporting_reads

        variants.sort(
            key=sort_key,
            reverse=True  # descending order of evidence.
        )

        yield from variants

    def construct_variants_using_assembly(self, nmf_lower_bound: float) -> Iterator[FloppStrainVariant]:
        """New implementation: Use NMF to separate the source strain abundances."""
        # marker_contig_counts = np.stack([
        #     contig_spec.contig.read_counts[contig_spec.strand_idx]
        #     for contig_spec in self.strand_specs
        # ], axis=1).sum(axis=0).reshape(-1, 1)  # (M x 1)
        #
        # _t, _m = marker_contig_counts.shape
        # # (by default, S = M)
        #
        # """NMF: decompose (M x 1) = (M x S) x (S x 1)"""
        # H = NMF(
        #     n_components=len(self.strand_specs),
        #     l1_ratio=0.0,
        #     random_state=0,
        #     max_iter=50000,
        #     alpha_W=0.1,
        #     # alpha_H=0.1
        # ).fit_transform(
        #     X=marker_contig_counts
        # )
        #
        # #  H is an (M x S) matrix.
        # for s in range(_m):
        #     strand_group = np.where(H[:, s] > nmf_lower_bound)[0]
        #     if len(strand_group) == 0:
        #         continue
        #
        #     yield self.strain_variant_from_grouping(
        #         strand_group,
        #         {assembly.canonical_marker: assembly for assembly in self.assemblies}
        #     )


        # OLD implementation: use glasso-estimated precision matrices.
        precision_matrix = self.compute_precision_matrix()
        partial_corrs = partial_corr_matrix(precision_matrix)
        rows, cols = upper_triangular_bounded(
            partial_corrs,
            k=1,
            upper_bound=np.inf,
            lower_bound=0
        )

        G = nx.Graph()
        for k in range(precision_matrix.shape[0]):
            G.add_node(k, strand=self.strand_specs[k])
        for v, w in zip(rows, cols):
            G.add_edge(v, w)

        for clique in nx.find_cliques(G):
            logger.debug("Clique: [{}]".format(
                ",".join(
                    str(self.strand_specs[x]) for x in clique
                )
            ))
            yield self.strain_variant_from_clique(
                G,
                clique,
                {assembly.canonical_marker: assembly for assembly in self.assemblies}
            )

        # # OLD implementation v2: use linear regression to uncover linear dependencies.
        # precision_matrix = self.compute_precision_matrix()
        # partial_corrs = partial_corr_matrix(precision_matrix)
        # rows, cols = upper_triangular_bounded(
        #     partial_corrs,
        #     k=1,
        #     upper_bound=np.inf,
        #     lower_bound=partial_corr_lower_bound
        # )
        #
        # G = nx.Graph()
        # for k in range(precision_matrix.shape[0]):
        #     G.add_node(k, strand=self.strand_specs[k])
        # for v, w in zip(rows, cols):
        #     G.add_edge(v, w)
        #
        # for clique in nx.find_cliques(G):
        #     logger.debug("Clique: [{}]".format(
        #         ",".join(
        #             str(self.strand_specs[x]) for x in clique
        #         )
        #     ))
        #     yield self.strain_variant_from_clique(
        #         G,
        #         clique,
        #         {assembly.canonical_marker: assembly for assembly in self.assemblies}
        #     )


    def strain_variant_from_grouping(self,
                                     strand_grouping: List[int],
                                     marker_to_assembly: Dict[Marker, FloppMarkerAssembly]
                                     ) -> FloppStrainVariant:
        # Group together the nodes by their corresponding marker.
        markers_to_subcliques: Dict[Marker, List[GloppContigStrandSpecification]] = defaultdict(list)
        for k in strand_grouping:
            strand: GloppContigStrandSpecification = self.strand_specs[k]
            markers_to_subcliques[strand.contig.canonical_marker].append(strand)

        marker_variants: List[FloppMarkerVariant] = []
        for marker, subclique in markers_to_subcliques.items():
            marker_variants += list(
                self.marker_variants_from_clique(marker, marker_to_assembly[marker], subclique)
            )

        # Determine the base strain using the best-matching marker.
        base_strain = self.db.best_matching_strain([
            marker_variant.base_marker for marker_variant in marker_variants
        ])

        variant_id = "{}<{}>".format(
            base_strain.id,
            ",".join(
                variant.id for variant in marker_variants
            )
        )

        logger.debug("Creating Strain Variant ({})".format(
            variant_id
        ))

        return FloppStrainVariant(
            base_strain=base_strain,
            id=variant_id,
            variant_markers=marker_variants
        )

    def compute_precision_matrix(self) -> np.ndarray:
        # marker_contig_counts = np.stack([
        #     contig_spec.contig.mean_counts[contig_spec.strand_idx]
        #     for contig_spec in self.strand_specs
        # ], axis=1)  # (T x M)

        marker_contig_counts = np.stack([
            contig_spec.contig.read_counts[contig_spec.strand_idx]
            for contig_spec in self.strand_specs
        ], axis=1)  # (T x M)

        # For now, just work with counts, assuming constant read depth (otherwise, might get spurious correlations).
        # NOTE: this doesn't normalize by read depths. However, note that dividing by read depths makes this ill-conditioned.
        #  Instead, divide by (relative read depth), e.g. the ratio read_depth_t / read_depth_1

        # marker_contig_counts = marker_contig_counts / np.array([
        #     time_slice.read_depth
        #     for time_slice in reads
        # ]).reshape([1, -1])

        return self.run_glasso(marker_contig_counts)

    def run_glasso(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: An (N x M) array, N = # of samples, M = # of features.
        :return: The estimated (M x M) precision matrix.
        """
        if self.glasso_standardize:
            myScaler = preprocessing.StandardScaler()
            X = myScaler.fit_transform(X)

        emp_cov = covariance.empirical_covariance(X)

        # Set shrinkage closer to 1 for poorly-conditioned data
        shrunk_cov = covariance.shrunk_covariance(emp_cov, shrinkage=self.glasso_shrinkage)

        _, precision = covariance.graphical_lasso(
            shrunk_cov,
            self.glasso_alpha,
            max_iter=self.glasso_iterations,
            tol=self.glasso_tol
        )
        logger.debug("Calculated graph-lasso covariance matrix for alpha={:.2e}".format(self.glasso_alpha))
        return precision

    def strain_variant_from_clique(self,
                                   G: nx.Graph,
                                   clique: List[int],
                                   marker_to_assembly: Dict[Marker, FloppMarkerAssembly]
                                   ) -> FloppStrainVariant:
        # Group together the nodes by their corresponding marker.
        markers_to_subcliques: Dict[Marker, List[GloppContigStrandSpecification]] = defaultdict(list)
        for v in clique:
            node_data: GloppContigStrandSpecification = G.nodes[v]['strand']
            markers_to_subcliques[node_data.contig.canonical_marker].append(node_data)

        marker_variants: List[FloppMarkerVariant] = []
        for marker, subclique in markers_to_subcliques.items():
            marker_variants += list(
                self.marker_variants_from_clique(marker, marker_to_assembly[marker], subclique)
            )

        # Determine the base strain using the best-matching marker.
        base_strain = self.db.best_matching_strain([
            marker_variant.base_marker for marker_variant in marker_variants
        ])

        variant_id = "{}<{}>".format(
            base_strain.id,
            ",".join(
                variant.id for variant in marker_variants
            )
        )

        logger.debug("Creating Strain Variant ({})".format(
            variant_id
        ))

        return FloppStrainVariant(
            base_strain=base_strain,
            id=variant_id,
            variant_markers=marker_variants
        )

    def marker_variants_from_clique(self,
                                    base_marker: Marker,
                                    marker_assembly: FloppMarkerAssembly,
                                    clique: List[GloppContigStrandSpecification]
                                    ) -> Iterator[AbstractMarkerVariant]:
        # for each contig index, tally up with variant assemblies it has, if any.
        variants_by_contig: List[List[GloppContigStrandSpecification]] = [
            [] for _ in range(marker_assembly.num_contigs)
        ]
        for spec in clique:
            assert spec.contig.canonical_marker.id == base_marker.id
            variants_by_contig[spec.contig.contig_idx].append(spec)
            if len(variants_by_contig[spec.contig.contig_idx]) > 1:
                logger.warning(
                    f"Corr grouping has multiple strands for contig {spec.contig.contig_idx} of {base_marker.id}. "
                    "Default behavior is to use combinatorial combinations."
                )

        """
        Take all possible combinatorial combinations.
        - marker_strands represents a choice of a strand for each contig in the assembly.
        - If there is no strand available in the grouping, then we pass "Base" as an indicator
        to take the reference seq.
        """
        for marker_strands in itertools.product(
            *(
                contig_arr
                if len(contig_arr) > 0
                else ["Base"]
                for contig_idx, contig_arr in enumerate(variants_by_contig)
            )
        ):
            yield self.create_marker_variant(base_marker, marker_assembly, marker_strands)

    @staticmethod
    def create_marker_variant(
            base_marker: Marker,
            marker_assembly: FloppMarkerAssembly,
            strands: Iterable[Union[GloppContigStrandSpecification, str]]
    ) -> AbstractMarkerVariant:
        seq_with_gaps, read_count = marker_assembly.contig_base_seq([
            None if isinstance(strand, str) else strand.strand_idx
            for strand in strands
        ])
        variant_id = "{}:Strands({})".format(
            base_marker.id,
            ".".join(
                "*" if isinstance(strand, str) else str(strand.strand_idx)
                for strand in strands
            ),
        )
        logger.debug("Creating Marker Variant ({})".format(
            variant_id
        ))
        return FloppMarkerVariant(
            id=variant_id,
            base_marker=base_marker,
            seq_with_gaps=seq_with_gaps,
            aln=marker_assembly.aln,
            num_supporting_reads=read_count,
            contig_strands=[None if isinstance(strand, str) else strand.strand_idx for strand in strands]
        )
