"""
Contains alignment-specific subroutines necessary for other algorithm implementations.
"""

from pathlib import Path
from typing import Iterable, Optional, Dict, List

from chronostrain.algs.subroutines.read_cache import ReadsComputationCache
from chronostrain.model import Marker
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.data_cache import ComputationCache
from chronostrain.util.external.bwa import bwa_mem, bwa_index
from chronostrain.util.alignments import parse_alignments, SequenceReadAlignment, SamHandler
from chronostrain.database import StrainDatabase


class CachedReadAlignments(object):
    """
    A wrapper around bwa_mem and bwa_index, but checks whether the output of these alignments already exist.
    If so, load them from disk instead of re-computing them.
    """
    def __init__(self,
                 reads: TimeSeriesReads,
                 db: StrainDatabase,
                 cache_override: Optional[ComputationCache] = None):
        self.reads = reads
        self.db = db
        self.marker_reference_path = db.multifasta_file

        if cache_override is not None:
            self.cache = cache_override
        else:
            self.cache = ReadsComputationCache(reads)
        bwa_index(self.marker_reference_path)

    @staticmethod
    def get_path(reads_path: Path) -> Path:
        return Path("alignments") / "{}.sam".format(reads_path.stem)

    def get_alignments(self, t_idx: int) -> Dict[Marker, List[SequenceReadAlignment]]:
        time_slice = self.reads[t_idx]
        alignments = {
            marker: []
            for marker in self.db.all_markers()
        }
        for reads_path in time_slice.src.paths:
            handler = self._get_alignment(reads_path, time_slice.src.quality_format)
            for marker, alns in parse_alignments(handler, self.db).items():
                alignments[marker] = alignments[marker] + alns
        return alignments

    def _get_alignment(self, reads_path: Path, quality_format: str) -> SamHandler:
        # ====== function bindings to pass to ComputationCache.
        def perform_alignment(align_path: Path, ref_path: Path, reads_path: Path):
            align_path.parent.mkdir(exist_ok=True, parents=True)
            bwa_mem(
                output_path=align_path,
                reference_path=ref_path,
                read_path=reads_path,
                min_seed_length=20,
                report_all_alignments=True
            )
            return SamHandler(align_path, ref_path, quality_format)

        # ====== Run the cached computation.
        alignment_output_path = self.cache.cache_dir / self.get_path(reads_path)
        return self.cache.call(
            filename=alignment_output_path,
            fn=perform_alignment,
            save=lambda p, o: None,
            load=lambda p: SamHandler(alignment_output_path, self.marker_reference_path, quality_format),
            kwargs={
                "align_path": alignment_output_path,
                "ref_path": self.marker_reference_path,
                "reads_path": reads_path
            }
        )
