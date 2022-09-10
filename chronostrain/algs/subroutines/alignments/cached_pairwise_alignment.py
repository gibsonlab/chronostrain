"""
Contains alignment-specific subroutines necessary for other algorithm implementations.
"""

from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import numpy as np

from chronostrain.algs.subroutines.cache import ReadsComputationCache
from chronostrain.model import Marker
from chronostrain.model.io import TimeSeriesReads, TimeSliceReadSource, ReadType
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

    def get_aligner(self, read_src: TimeSliceReadSource):
        if cfg.external_tools_cfg.pairwise_align_cmd == "ssw-align":
            return SmithWatermanAligner(
                reference_path=self.marker_reference_path,
                match_score=np.floor((4 * 500) / self.reads.min_read_length).astype(int),
                mismatch_penalty=np.floor(np.log(10) * 4.2),
                gap_open_penalty=np.mean(
                    [-cfg.model_cfg.get_float("INSERTION_LL_1"), -cfg.model_cfg.get_float("DELETION_LL_1")]
                ).astype(int).item(),
                gap_extend_penalty=0,
                score_threshold=1,
            )
        elif cfg.external_tools_cfg.pairwise_align_cmd == "bwa":
            if read_src.read_type == ReadType.PAIRED_END_1:
                insertion_ll = cfg.model_cfg.get_float("INSERTION_LL_1")
                deletion_ll = cfg.model_cfg.get_float("DELETION_LL_1")
            elif read_src.read_type == ReadType.PAIRED_END_2:
                insertion_ll = cfg.model_cfg.get_float("INSERTION_LL_2")
                deletion_ll = cfg.model_cfg.get_float("DELETION_LL_2")
            elif read_src.read_type == ReadType.SINGLE_END:
                insertion_ll = cfg.model_cfg.get_float("INSERTION_LL")
                deletion_ll = cfg.model_cfg.get_float("DELETION_LL")
            else:
                raise NotImplementedError(f"Read type `{read_src.read_type}` not implemented for aligner.")

            return BwaAligner(
                reference_path=self.db.multifasta_file,
                min_seed_len=15,
                reseed_ratio=0.2,  # default; smaller = slower but more alignments.
                bandwidth=10,
                num_threads=self.num_cores,
                report_all_alignments=True,
                match_score=2,  # log likelihood ratio log_2(4p)
                mismatch_penalty=5,  # Assume quality score of 20, log likelihood ratio log_2(4 * error * <3/4>)
                off_diag_dropoff=100,
                gap_open_penalty=(0, 0),
                gap_extend_penalty=(
                    int(-deletion_ll / np.log(2)),
                    int(-insertion_ll / np.log(2))
                ),
                clip_penalty=1,
                score_threshold=50,
                bwa_command='bwa'
            )
        elif cfg.external_tools_cfg.pairwise_align_cmd == "bowtie2":
            return BowtieAligner(
                reference_path=self.marker_reference_path,
                index_basepath=self.marker_reference_path.parent,
                index_basename=self.marker_reference_path.stem,
                num_threads=self.num_cores,
                report_all_alignments=True,
                seed_length=22,
                seed_extend_failures=15,
                num_reseeds=5,
                score_min_fn=bt2_func_constant(const=50),
                score_match_bonus=2,
                score_mismatch_penalty=np.floor(
                    [5, 0]
                ).astype(int),
                score_read_gap_penalty=np.floor(
                    [0, int(-cfg.model_cfg.get_float("DELETION_LL_1") / np.log(2))]
                ).astype(int),
                score_ref_gap_penalty=np.floor(
                    [0, int(-cfg.model_cfg.get_float("INSERTION_LL_1") / np.log(2))]
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
            sam_file = self._get_alignment(src)
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
            sam_file = self._get_alignment(src)
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
                sam_file = self._get_alignment(src)
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
                sam_file = self._get_alignment(src)
                for aln in parse_alignments(
                        sam_file,
                        self.db,
                        read_getter=lambda read_id: time_slice.get_read(read_id),
                        reattach_clipped_bases=True
                ):
                    yield t_idx, aln

    def _get_alignment(self, read_src: TimeSliceReadSource) -> SamFile:
        # ====== Files relative to cache dir.
        cache_relative_path = Path("alignments") / self.get_path(read_src.path)
        absolute_path = self.cache.cache_dir / cache_relative_path

        # ====== function bindings to pass to ComputationCache.
        def perform_alignment():
            absolute_path.parent.mkdir(exist_ok=True, parents=True)
            self.get_aligner(read_src).align(
                query_path=read_src.path,
                output_path=absolute_path,
                read_type=read_src.read_type
            )
            return SamFile(absolute_path, read_src.quality_format)

        # ====== Run the cached computation.
        return self.cache.call(
            relative_filepath=cache_relative_path,
            fn=perform_alignment,
            save=lambda path, obj: None,
            load=lambda path: SamFile(path, read_src.quality_format)
        )
