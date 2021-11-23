"""
Contains alignment-specific subroutines necessary for other algorithm implementations.
"""

from pathlib import Path
from typing import Optional, Dict, List, Iterator, Tuple

from chronostrain.model import Marker
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.cache import ComputationCache
from chronostrain.util.alignments.sam import SamFile
from chronostrain.util.alignments.pairwise import *
from chronostrain.database import StrainDatabase

from chronostrain.algs.subroutines.cache import ReadsComputationCache
from chronostrain.config import cfg


class CachedReadPairwiseAlignments(object):
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

        if cfg.external_tools_cfg.pairwise_align_cmd == "bwa":
            self.aligner = BwaAligner(
                reference_path=self.marker_reference_path,
                min_seed_len=8,
                num_threads=cfg.model_cfg.num_cores,
                report_all_alignments=True
            )
        elif cfg.external_tools_cfg.pairwise_align_cmd == "bowtie2":
            self.aligner = BowtieAligner(
                reference_path=self.marker_reference_path,
                index_basepath=self.marker_reference_path.parent,
                index_basename="markers",
                num_threads=cfg.model_cfg.num_cores
            )
        else:
            raise NotImplementedError(
                f"Alignment command `{cfg.external_tools_cfg.pairwise_align_cmd}` not currently supported."
            )

    @staticmethod
    def get_path(reads_path: Path) -> Path:
        return Path("") / "{}.sam".format(reads_path.stem)

    def alignments_by_marker_and_timepoint(self, t_idx: int) -> Dict[Marker, List[SequenceReadPairwiseAlignment]]:
        """
        Returns a mapping of marker -> (read alignments to marker from t[t_idx])
        """
        time_slice = self.reads[t_idx]
        alignments = {
            marker: []
            for marker in self.db.all_markers()
        }
        for reads_path in time_slice.src.paths:
            sam_file = self._get_alignment(reads_path, time_slice.src.quality_format)
            for marker, alns in marker_categorized_alignments(
                    sam_file,
                    self.db,
                    lambda read_id: time_slice.get_read(read_id),
                    ignore_edge_mapped_reads=True
            ).items():
                alignments[marker] = alignments[marker] + alns
        return alignments

    def reads_with_alignments_to_marker(self) -> Iterator[Tuple[Marker, List[List[SequenceReadPairwiseAlignment]]]]:
        """
        Returns a mapping of marker -> (read alignments to marker from t, across all t_idx)
        """
        marker_to_reads: Dict[Marker, List[List[SequenceReadPairwiseAlignment]]] = {
            marker: [
                [] for _ in self.reads
            ] for marker in self.db.all_markers()
        }
        for t_idx, time_slice in enumerate(self.reads):
            for reads_path in time_slice.src.paths:
                sam_file = self._get_alignment(reads_path, time_slice.src.quality_format)
                for aln in parse_alignments(
                        sam_file,
                        self.db,
                        read_getter=lambda read_id: time_slice.get_read(read_id),
                ):
                    marker_to_reads[aln.marker][t_idx].append(aln)
        yield from marker_to_reads.items()

    def get_alignments(self) -> Iterator[Tuple[int, SequenceReadPairwiseAlignment]]:
        for t_idx, time_slice in enumerate(self.reads):
            for reads_path in time_slice.src.paths:
                sam_file = self._get_alignment(reads_path, time_slice.src.quality_format)
                for aln in parse_alignments(
                        sam_file,
                        self.db,
                        read_getter=lambda read_id: time_slice.get_read(read_id),
                ):
                    yield t_idx, aln

    def _get_alignment(self, reads_path: Path, quality_format: str) -> SamFile:
        # ====== Files relative to cache dir.
        cache_relative_path = Path("alignments") / self.get_path(reads_path)
        absolute_path = self.cache.cache_dir / cache_relative_path

        # ====== function bindings to pass to ComputationCache.
        def perform_alignment():
            absolute_path.parent.mkdir(exist_ok=True, parents=True)
            self.aligner.align(
                query_path=reads_path,
                output_path=absolute_path
            )
            return SamFile(absolute_path, quality_format)

        # ====== Run the cached computation.
        return self.cache.call(
            relative_filepath=cache_relative_path,
            fn=perform_alignment,
            save=lambda path, obj: None,
            load=lambda path: SamFile(path, quality_format)
        )
