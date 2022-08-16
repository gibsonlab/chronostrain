"""
    glopp_with_corr.py
    An algorithm which resolves strain variants (metagenomic "haplotypes") using a combination of glopp and time-series
    correlation grouping.
"""
import itertools
from collections import defaultdict
from typing import List, Dict, Iterator, Union, Iterable, Optional, Set, Tuple

from chronostrain.logging import create_logger
from chronostrain.database import StrainDatabase
from chronostrain.algs.subroutines.assembly import CachedGloppVariantAssembly
from chronostrain.model import AbstractMarkerVariant, StrainVariant
from .base import FloppMarkerVariant, FloppStrainVariant
from ...model import Marker
from ...model.io import TimeSeriesReads
from .meta import ExhaustiveVariantADVISolver
from ...util.flopp import FloppMarkerAssembly, FloppMarkerContig

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


class GloppExhaustiveVariantSolver(ExhaustiveVariantADVISolver):
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
                 # seed_with_database: bool = False,
                 num_cores: int = 1,
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
            bbvi_num_samples=bbvi_num_samples,
            num_cores=num_cores
            # seed_with_database=seed_with_database
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
        for strain_variant in used_variants:
            if not isinstance(strain_variant, FloppStrainVariant):
                raise TypeError(
                    "Passed strain variant must be of class {}".format(FloppStrainVariant.__class__.__name__)
                )

        # noinspection PyTypeChecker
        yield from self.strain_variant_from_grouping(
            used_variants,
            {assembly.canonical_marker: assembly for assembly in self.assemblies}
        )

    def strain_variant_from_grouping(self,
                                     used_variants: List[FloppStrainVariant],
                                     marker_to_assembly: Dict[Marker, FloppMarkerAssembly]
                                     ) -> FloppStrainVariant:
        # Take note of what strands were already used.
        used_strands: Dict[Marker, Set[Tuple[int, int]]] = {
            marker: set() for marker in marker_to_assembly.keys()
        }
        for strain in used_variants:
            for marker in strain.flopp_variant_markers:
                marker_used_strands = used_strands[marker.base_marker]
                for contig_idx, strand_idx in enumerate(marker.contig_strands):
                    if strand_idx is not None:
                        marker_used_strands.add((contig_idx, strand_idx))
                        print("DEBUG: excluding ({}, {})".format(contig_idx, strand_idx))

        # Group together the nodes by their corresponding marker.
        markers_to_subcliques: Dict[Marker, List[GloppContigStrandSpecification]] = defaultdict(list)
        for strand_spec in self.strand_specs:
            if (strand_spec.contig.contig_idx, strand_spec.strand_idx) not in used_strands[strand_spec.contig.canonical_marker]:
                markers_to_subcliques[strand_spec.contig.canonical_marker].append(strand_spec)

        for marker, subclique in markers_to_subcliques.items():
            for marker_variant in self.marker_variants_from_clique(
                    marker,
                    marker_to_assembly[marker],
                    subclique
            ):
                # Determine the base strain using the best-matching marker.
                base_strain = self.db.best_matching_strain([marker_variant.base_marker])

                variant_id = "{}<{}>".format(
                    base_strain.id,
                    marker_variant.id
                )

                logger.debug("Creating Strain Variant ({})".format(
                    variant_id
                ))

                yield FloppStrainVariant(
                    base_strain=base_strain,
                    id=variant_id,
                    variant_markers=[marker_variant]
                )

    def marker_variants_from_clique(self,
                                    base_marker: Marker,
                                    marker_assembly: FloppMarkerAssembly,
                                    clique: List[GloppContigStrandSpecification]
                                    ) -> Iterator[FloppMarkerVariant]:
        # for each contig index, tally up with variant assemblies it has, if any.
        variants_by_contig: List[List[GloppContigStrandSpecification]] = [
            [] for _ in range(marker_assembly.num_contigs)
        ]
        for spec in clique:
            assert spec.contig.canonical_marker.id == base_marker.id
            variants_by_contig[spec.contig.contig_idx].append(spec)
            # if len(variants_by_contig[spec.contig.contig_idx]) > 1:
            #     logger.warning(
            #         f"Corr grouping has multiple strands for contig {spec.contig.contig_idx} of {base_marker.id}. "
            #         "Default behavior is to use trivial assembly."
            #     )

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
