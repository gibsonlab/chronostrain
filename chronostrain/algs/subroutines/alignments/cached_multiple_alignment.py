"""
Contains alignment-specific subroutines necessary for other algorithm implementations.
"""

from pathlib import Path
from typing import Optional, List, Iterator, Tuple

from chronostrain.model import Marker
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.alignments.pairwise import SequenceReadPairwiseAlignment
from chronostrain.util.cache import ComputationCache
import chronostrain.util.alignments.multiple as multialign
from chronostrain.database import StrainDatabase

from chronostrain.algs.subroutines.cache import ReadsComputationCache
from .cached_pairwise_alignment import CachedReadPairwiseAlignments

from chronostrain.config import create_logger
logger = create_logger(__name__)


class CachedReadMultipleAlignments(object):
    def __init__(self,
                 reads: TimeSeriesReads,
                 db: StrainDatabase,
                 cache_override: Optional[ComputationCache] = None):
        self.reads = reads
        self.db = db

        if cache_override is not None:
            self.cache = cache_override
        else:
            self.cache = ReadsComputationCache(reads)

    @staticmethod
    def get_path(reads_path: Path) -> Path:
        return Path("") / "{}.sam".format(reads_path.stem)

    def pairwise_seed(self) -> Iterator[Tuple[Marker, List[List[SequenceReadPairwiseAlignment]]]]:
        cache_pairwise_align = CachedReadPairwiseAlignments(self.reads, self.db, cache_override=self.cache)
        yield from cache_pairwise_align.reads_with_alignments_to_marker()

    def get_alignments(self) -> Iterator[multialign.MarkerMultipleFragmentAlignment]:
        """
        Uses a pairwise alignment to initialize a multiple alignment.

        The "initialization" simply refers to the usage of the pairwise align to determine all
        reads which map to particular markers (one by one).
        For each marker, a multiple alignment is performed with the collection reads that
        map to it.
        """
        for m_idx, (marker, timeseries_pairwise_aligns) in enumerate(self.pairwise_seed()):
            logger.debug("Computing multiple alignment for marker `{}`... ({} of {})".format(
                marker.id,
                m_idx + 1,
                self.db.num_markers()
            ))
            yield self._perform_cached_alignment(marker, timeseries_pairwise_aligns)

    def _perform_cached_alignment(self,
                                  marker: Marker,
                                  timeseries_pairwise_aligns: List[List[SequenceReadPairwiseAlignment]]
                                  ) -> multialign.MarkerMultipleFragmentAlignment:
        def read_gen():
            for t_idx, alignments in enumerate(timeseries_pairwise_aligns):
                for aln in alignments:
                    yield t_idx, aln.read, aln.reverse_complemented

        # ====== function bindings to pass to ComputationCache.
        def perform_alignment(out_path: Path) -> multialign.MarkerMultipleFragmentAlignment:
            base_dir = out_path.parent
            base_dir.mkdir(exist_ok=True, parents=True)
            multialign.align(
                marker=marker,
                read_descriptions=read_gen(),
                intermediate_fasta_path=base_dir / f"{marker.id}_reads.fasta",
                out_fasta_path=out_path
            )
            return multialign.parse(marker, self.reads, out_path)

        # ====== Run the cached computation.
        cache_relative_path = Path("multiple_alignments") / f"{marker.id}_multi_align.fasta"

        return self.cache.call(
            relative_filepath=cache_relative_path,
            fn=perform_alignment,
            save=lambda path, obj: None,
            load=lambda path: multialign.parse(marker, self.reads, path),
            call_kwargs={
                "out_path": self.cache.cache_dir / cache_relative_path
            }
        )
