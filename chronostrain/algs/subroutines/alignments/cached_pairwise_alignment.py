"""
Contains alignment-specific subroutines necessary for other algorithm implementations.
"""

from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import numpy as np

from chronostrain.algs.subroutines.cache import ReadsComputationCache
from chronostrain.model import Marker
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.alignments.sam import SamFile
from chronostrain.util.alignments.pairwise import *
from chronostrain.database import StrainDatabase

from chronostrain.config import cfg
from chronostrain.util.external import bt2_func_constant


class CachedReadPairwiseAlignments(object):
    """
    A wrapper around bwa_mem and bwa_index, but checks whether the output of these alignments already exist.
    If so, load them from disk instead of re-computing them.
    """
    def __init__(self,
                 reads: TimeSeriesReads,
                 db: StrainDatabase,
                 num_cores: int = 1):
        self.reads = reads
        self.db = db
        self.num_cores = num_cores
        self.marker_reference_path = db.multifasta_file
        self.cache = ReadsComputationCache(reads)

        if cfg.external_tools_cfg.pairwise_align_cmd == "ssw-align":
            self.aligner = SmithWatermanAligner(
                reference_path=self.marker_reference_path,
                match_score=np.floor((4 * 500) / reads.min_read_length).astype(int),
                mismatch_penalty=np.floor(np.log(10) * 4.2),
                gap_open_penalty=np.mean(
                    [-cfg.model_cfg.get_float("INSERTION_LL_1"), -cfg.model_cfg.get_float("DELETION_LL_1")]
                ).astype(int).item(),
                gap_extend_penalty=0,
                score_threshold=1,
            )
        elif cfg.external_tools_cfg.pairwise_align_cmd == "bwa":
            self.aligner = BwaAligner(
                reference_path=self.db.multifasta_file,
                min_seed_len=15,
                reseed_ratio=1,  # default; smaller = slower but more alignments.
                bandwidth=10,
                num_threads=self.num_cores,
                report_all_alignments=True,
                match_score=2,  # log likelihood ratio log_2(4p)
                mismatch_penalty=5,  # Assume quality score of 20, log likelihood ratio log_2(4 * error * <3/4>)
                off_diag_dropoff=100,
                gap_open_penalty=(0, 0),
                gap_extend_penalty=(0, 0),
                clip_penalty=5,
                score_threshold=50
            )
        elif cfg.external_tools_cfg.pairwise_align_cmd == "bowtie2":
            self.aligner = BowtieAligner(
                reference_path=self.marker_reference_path,
                index_basepath=self.marker_reference_path.parent,
                index_basename=self.marker_reference_path.stem,
                num_threads=self.num_cores,
                report_all_alignments=True,
                num_reseeds=22,
                score_min_fn=bt2_func_constant(const=-500),
                score_mismatch_penalty=np.floor(
                    [np.log(3) + 4 * np.log(10), 0]
                ).astype(int),
                score_read_gap_penalty=np.floor(
                    [0, -cfg.model_cfg.get_float("INSERTION_LL_1")]
                ).astype(int),
                score_ref_gap_penalty=np.floor(
                    [0, -cfg.model_cfg.get_float("DELETION_LL_1")]
                ).astype(int)
            )
        else:
            raise NotImplementedError(
                f"Alignment command `{cfg.external_tools_cfg.pairwise_align_cmd}` not currently supported."
            )

    @staticmethod
    def get_path(reads_path: Path) -> Path:
        return Path("") / "{}.sam".format(reads_path.stem)

    def alignments_by_timepoint(self, t_idx: int) -> Iterator[SequenceReadPairwiseAlignment]:
        time_slice = self.reads[t_idx]
        for src in time_slice.sources:
            sam_file = self._get_alignment(src.path, src.quality_format)
            yield from parse_alignments(
                sam_file,
                self.db,
                read_getter=lambda read_id: time_slice.get_read(read_id),
                reattach_clipped_bases=True,
                min_hit_ratio=0.50,
                min_frag_len=15
            )

    def alignments_by_marker_and_timepoint(self, t_idx: int) -> Iterator[Tuple[Marker, List[SequenceReadPairwiseAlignment]]]:
        """
        DEPRECATED.
        Returns a mapping of marker -> (read alignments to marker from t[t_idx])
        """
        time_slice = self.reads[t_idx]
        for src in time_slice.sources:
            sam_file = self._get_alignment(src.path, src.quality_format)
            for marker, alns in marker_categorized_alignments(
                    sam_file,
                    self.db,
                    read_getter=lambda read_id: time_slice.get_read(read_id),
                    reattach_clipped_bases=True,
                    min_hit_ratio=0.50,
                    min_frag_len=15
            ).items():
                yield marker, alns

    def reads_with_alignments_to_marker(self) -> Iterator[Tuple[Marker, List[List[SequenceReadPairwiseAlignment]]]]:
        """
        DEPRECATED.
        Returns a mapping of marker -> (read alignments to marker from t, across all t_idx)
        """
        marker_to_reads: Dict[Marker, List[List[SequenceReadPairwiseAlignment]]] = {
            marker: [
                [] for _ in self.reads
            ] for marker in self.db.all_markers()
        }
        for t_idx, time_slice in enumerate(self.reads):
            for src in time_slice.sources:
                sam_file = self._get_alignment(src.path, src.quality_format)
                for aln in parse_alignments(
                        sam_file,
                        self.db,
                        read_getter=lambda read_id: time_slice.get_read(read_id),
                        reattach_clipped_bases=True
                ):
                    marker_to_reads[aln.marker][t_idx].append(aln)
        yield from marker_to_reads.items()

    def get_alignments(self) -> Iterator[Tuple[int, SequenceReadPairwiseAlignment]]:
        for t_idx, time_slice in enumerate(self.reads):
            for src in time_slice.sources:
                sam_file = self._get_alignment(src.path, src.quality_format)
                for aln in parse_alignments(
                        sam_file,
                        self.db,
                        read_getter=lambda read_id: time_slice.get_read(read_id),
                        reattach_clipped_bases=True
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
