"""
    glopp_with_corr.py
    An algorithm which resolves strain variants (metagenomic "haplotypes") using a combination of glopp and time-series
    correlation grouping.
"""
from typing import List, Optional, Tuple

from chronostrain.config import create_logger
from chronostrain.database import StrainDatabase
from .base import StrainVariant
from ..subroutines import CachedReadMultipleAlignments, CachedGloppVariantAssembly
from ..subroutines.assembly import MarkerContig
from ...model import Population
from ...model.io import TimeSeriesReads
from ...util.alignments.multiple import MarkerMultipleFragmentAlignment

logger = create_logger(__name__)


def construct_variants(
        db: StrainDatabase,
        reads: TimeSeriesReads,
        quality_lower_bound: float,
        seed_with_database: Optional[bool] = False
) -> Population:
    """
    In increasing order of size, computes a group of strain variants which explain the specified input reads.

    :param db: The database of markers and strains.
    :param reads: The input time-series reads.
    :param quality_lower_bound: The lower bound at which to apply a per-nucleotide filter for variant calling.
    :param seed_with_database: Indicates whether or not to include the database reference strains (for test cases,
    where reads were actually sampled from the database strains.
    :return: A strain population instance, which may or may not include StrainVariant instances (depending on whether
    seed_with_database was set, and on whether any improvement is made in the log-likelihood).
    """
    marker_alignments = list(CachedReadMultipleAlignments(reads, db).get_alignments())

    if seed_with_database:
        raise NotImplementedError("seed_with_database not implemented yet.")

    n_max_variants = 100  # TODO: set this in a more smart way. Maybe count the number of single nucleotide variants from an alignment?
    best_pop: Population = None
    best_ll: float = float("-inf")
    for n_variants in range(1, n_max_variants + 1):
        new_ll, new_pop = construct_variants_with_ploidy(
            db,
            reads,
            marker_alignments,
            quality_lower_bound,
            n_variants
        )
        if new_ll > best_ll:
            logger.debug(
                "Found data log-likelihood improvement from {} variants to {} variants ({:.3f} -> {:.3f})".format(
                    n_variants - 1, n_variants,
                    best_ll, new_ll
                )
            )
            best_pop = new_pop
            best_ll = new_ll
        else:
            return best_pop
    raise RuntimeError(f"Unable to find an optimal population within {n_max_variants} variant upper bound.")


def construct_variants_with_ploidy(db: StrainDatabase,
                                   reads: TimeSeriesReads,
                                   alns: List[MarkerMultipleFragmentAlignment],
                                   q_lower_bound: float,
                                   ploidy: int
                                   ) -> Tuple[float, Population]:
    marker_ordering = [aln.marker for aln in alns]
    num_contig_per_marker = []

    for marker_multi_align in alns:
        marker_contigs: List[MarkerContig] = CachedGloppVariantAssembly(
            reads, marker_multi_align
        ).run_glopp(num_variants=ploidy)

