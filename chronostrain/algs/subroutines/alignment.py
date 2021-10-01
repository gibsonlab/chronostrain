"""
Contains alignment-specific subroutines necessary for other algorithm implementations.
"""

from pathlib import Path
from typing import Iterable, Optional

from chronostrain.algs.subroutines.read_cache import ReadsComputationCache
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.data_cache import ComputationCache
from chronostrain.util.external.bwa import bwa_mem, bwa_index
from chronostrain.util.sam_handler import SamHandler


class CachedReadAlignments(object):
    """
    A wrapper around bwa_mem and bwa_index, but checks whether the output of these alignments already exist.
    If so, load them from disk instead of re-computing them.
    """
    def __init__(self, marker_reference_path: Path,
                 reads: TimeSeriesReads,
                 cache_override: Optional[ComputationCache] = None):
        self.marker_reference_path = marker_reference_path
        self.reads = reads
        if cache_override is not None:
            self.cache = cache_override
        else:
            self.cache = ReadsComputationCache(reads)
        bwa_index(self.marker_reference_path)

    @staticmethod
    def get_path(reads_path: Path) -> Path:
        return Path("alignments") / "{}.sam".format(reads_path.stem)

    def get_alignments(self, t_idx: int) -> Iterable[SamHandler]:
        time_slice = self.reads[t_idx]
        for reads_path in self.reads[t_idx].src.paths:
            yield self._get_alignment(reads_path, time_slice.src.quality_format)

    def _get_alignment(self, reads_path: Path, quality_format: str) -> SamHandler:
        # ====== function bindings
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
