"""
Contains alignment-specific subroutines necessary for other algorithm implementations.
"""
from pathlib import Path
from typing import Optional, List, Iterator, Tuple, Dict

import chronostrain.util.alignments.multiple as multialign
from chronostrain.algs.subroutines.cache import ReadsComputationCache
from chronostrain.database import StrainDatabase
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.alignments.pairwise import SequenceReadPairwiseAlignment
from chronostrain.util.cache import ComputationCache
from .cached_pairwise_alignment import CachedReadPairwiseAlignments

from chronostrain.config import create_logger, cfg
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

    def pairwise_seed(self) -> Dict[str, List[List[SequenceReadPairwiseAlignment]]]:
        cache_pairwise_align = CachedReadPairwiseAlignments(self.reads, self.db, cache_override=self.cache)
        timeseries_alns_by_marker_name: Dict[str, List[List[SequenceReadPairwiseAlignment]]] = {
            marker.name: [
                [] for _ in range(len(self.reads))
            ]
            for marker in self.db.all_canonical_markers()
        }

        for t_idx, aln in cache_pairwise_align.get_alignments():
            try:
                timeseries_alns_by_marker_name[aln.marker.name][t_idx].append(aln)
            except KeyError:
                raise KeyError("Canonical marker of `{}` not found.".format(aln.marker.name)) from None

        return timeseries_alns_by_marker_name

    def get_alignments(self, num_cores: int = 1) -> Iterator[multialign.MarkerMultipleFragmentAlignment]:
        """
        Uses a pairwise alignment to initialize a multiple alignment.

        The "initialization" simply refers to the usage of the pairwise align to determine all
        reads which map to particular markers (one by one).
        For each marker, a multiple alignment is performed with the collection reads that
        map to it.
        """
        pairwise = self.pairwise_seed()
        for m_idx, (marker_name, timeseries_pairwise_alns) in enumerate(pairwise.items()):
            logger.debug("Computing multiple alignment for marker `{}`... ({} of {})".format(
                marker_name,
                m_idx + 1,
                len(pairwise)
            ))
            yield self._perform_cached_alignment(marker_name, timeseries_pairwise_alns, num_cores=num_cores)

    def _perform_cached_alignment(self,
                                  marker_name: str,
                                  timeseries_pairwise_aligns: List[List[SequenceReadPairwiseAlignment]],
                                  num_cores: int = 1,
                                  ) -> multialign.MarkerMultipleFragmentAlignment:
        def read_gen():
            seen_reads = set()
            alignments_with_time: List[Tuple[SequenceReadPairwiseAlignment, int]] = []
            for t_idx, alignments in enumerate(timeseries_pairwise_aligns):
                for aln in alignments:
                    if aln.read.id in seen_reads:
                        continue
                    alignments_with_time.append((aln, t_idx))
                    seen_reads.add(aln.read.id)

            alignments = sorted(alignments_with_time, key=lambda x: x[0].marker_start)
            for aln, t_idx in alignments:
                yield t_idx, aln.read, aln.reverse_complemented

        # ====== function bindings to pass to ComputationCache.
        def perform_alignment(out_path: Path) -> multialign.MarkerMultipleFragmentAlignment:
            base_dir = out_path.parent
            base_dir.mkdir(exist_ok=True, parents=True)
            multialign.align(
                db=self.db,
                marker_name=marker_name,
                read_descriptions=read_gen(),
                intermediate_fasta_path=base_dir / f"{marker_name}_reads.fasta",
                out_fasta_path=out_path,
                n_threads=num_cores
            )
            return multialign.parse(self.db, marker_name, self.reads, out_path)

        # ====== Run the cached computation.
        cache_relative_path = Path("multiple_alignments") / f"{marker_name}_multi_align.fasta"

        return self.cache.call(
            relative_filepath=cache_relative_path,
            fn=perform_alignment,
            save=lambda path, obj: None,
            load=lambda path: multialign.parse(self.db, marker_name, self.reads, path),
            call_kwargs={
                "out_path": self.cache.cache_dir / cache_relative_path
            }
        )
