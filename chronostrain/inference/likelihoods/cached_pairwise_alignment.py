"""
Contains alignment-specific likelihoods necessary for other algorithm implementations.
"""
import shutil
from typing import Iterator, Union, Callable, Tuple
from pathlib import Path
import numpy as np

from chronostrain.model import SequenceRead
from chronostrain.model.io import *
from chronostrain.util.alignments.sam import SamFile
from chronostrain.util.alignments.pairwise import *
from chronostrain.database import StrainDatabase

from chronostrain.config import cfg
from chronostrain.util.cache import ComputationCache
from chronostrain.util.external import bt2_func_linear


class CachedReadPairwiseAlignments(object):
    """
    A wrapper around bwa_mem and bwa_index, but checks whether the output of these alignments already exist.
    If so, load them from disk instead of re-computing them.
    """
    def __init__(self,
                 reads: TimeSeriesReads,
                 db: StrainDatabase,
                 cache: ComputationCache,
                 n_threads: int = 1,
                 print_tqdm: bool = True):
        self.reads = reads
        self.db = db
        self.n_threads = n_threads
        self.marker_reference_path = db.multifasta_file
        self.print_tqdm = print_tqdm
        self.cache = cache

        # Lazy initializations
        self.pe_forward_aligner: Union[AbstractPairwiseAligner, None] = None
        self.pe_reverse_aligner: Union[AbstractPairwiseAligner, None] = None
        self.single_end_aligner: Union[AbstractPairwiseAligner, None] = None

    def alignment_parameters(self, insertion_ll: float, deletion_ll: float) -> Tuple[int, int, int, int]:
        """
        Returns the 4-tuple consisting of:
        1) match_bonus
        2) mismatch_penalty (a positive number)
        3) insertion_penalty (a positive number)
        4) deletion_penalty (a positive number)
        """
        return 2, 5, int(-insertion_ll / np.log(2)), int(-deletion_ll / np.log(2))

    def get_pe_forward_aligner(self) -> AbstractPairwiseAligner:
        if self.pe_forward_aligner is None:
            match, mismatch, insertion, deletion = self.alignment_parameters(
                cfg.model_cfg.get_float("INSERTION_LL_1"),
                cfg.model_cfg.get_float("DELETION_LL_1")
            )
            self.pe_forward_aligner = self.get_parametrized_aligner(match, mismatch, insertion, deletion)
        return self.pe_forward_aligner

    def get_pe_reverse_aligner(self) -> AbstractPairwiseAligner:
        if self.pe_reverse_aligner is None:
            match, mismatch, insertion, deletion = self.alignment_parameters(
                cfg.model_cfg.get_float("INSERTION_LL_2"),
                cfg.model_cfg.get_float("DELETION_LL_2")
            )
            self.pe_reverse_aligner = self.get_parametrized_aligner(match, mismatch, insertion, deletion)
        return self.pe_reverse_aligner

    def get_singleend_aligner(self) -> AbstractPairwiseAligner:
        if self.single_end_aligner is None:
            match, mismatch, insertion, deletion = self.alignment_parameters(
                cfg.model_cfg.get_float("INSERTION_LL"),
                cfg.model_cfg.get_float("DELETION_LL")
            )
            self.single_end_aligner = self.get_parametrized_aligner(match, mismatch, insertion, deletion)
        return self.single_end_aligner

    def get_parametrized_aligner(
            self,
            match_bonus: int,
            mismatch_penalty: int,
            insertion_penalty: int,
            deletion_penalty: int,
    ) -> AbstractPairwiseAligner:
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
        elif cfg.external_tools_cfg.pairwise_align_cmd == "bwa" or cfg.external_tools_cfg.pairwise_align_cmd == "bwa-mem2":
            return BwaAligner(
                reference_path=self.db.multifasta_file,
                min_seed_len=10,
                reseed_ratio=0.5,  # smaller = slower but more alignments.
                mem_discard_threshold=90000,  # default is 50000
                chain_drop_threshold=0.1,  # default is 0.5
                bandwidth=10,
                num_threads=self.n_threads,
                report_all_alignments=True,
                match_score=match_bonus,  # log likelihood ratio log_2(4p)
                mismatch_penalty=mismatch_penalty,  # Assume quality score of 20, log likelihood ratio log_2(4 * error * <3/4>)
                off_diag_dropoff=100,
                gap_open_penalty=(0, 0),
                gap_extend_penalty=(deletion_penalty, insertion_penalty),
                clip_penalty=0,
                score_threshold=50,
                bwa_command=cfg.external_tools_cfg.pairwise_align_cmd
            )
        elif cfg.external_tools_cfg.pairwise_align_cmd == "bowtie2":
            return BowtieAligner(
                reference_path=self.marker_reference_path,
                index_basepath=self.marker_reference_path.parent,
                index_basename=self.marker_reference_path.stem,
                num_threads=self.n_threads,
                report_all_alignments=True,
                seed_length=22,
                seed_extend_failures=15,
                num_reseeds=5,
                score_min_fn=bt2_func_linear(const=0., coef=1.0),  # Ideally, want f(x)=2x-50 (so f(150) = match * x / 2) but bowtie2 doesn't allow functions that can be negative (if match_bonus > 0). So instead interpolate function using f(x)=Ax instead.
                # score_min_fn=bt2_func_constant(const=75),
                # score_min_fn=bt2_func_sqrt(const=0., coef=16),  # wiggle room is probably somewhere around 50 less than the maximal possible score, but bowtie2 doesn't allow functions that can be negative (if match_bonus > 0). This is kind of a dumb restriction, so let's approximate it using the function 16 * sqrt(x).
                score_match_bonus=2,
                score_mismatch_penalty=np.floor([mismatch_penalty, 0]).astype(int),
                score_read_gap_penalty=np.floor([0, deletion_penalty]).astype(int),
                score_ref_gap_penalty=np.floor([0, insertion_penalty]).astype(int)
            )
        else:
            raise NotImplementedError(
                f"Alignment command `{cfg.external_tools_cfg.pairwise_align_cmd}` not currently supported."
            )

    def align_single_end(self, path: Path, quality_format: str, read_getter: Callable[[str], SequenceRead]) -> Iterator[SequenceReadPairwiseAlignment]:
        aligner = self.get_singleend_aligner()
        sam_file = self.perform_alignment(path, ReadType.SINGLE_END, quality_format, aligner)
        yield from self.parse_alignments(sam_file, read_getter)

    def align_pe_forward(self, path: Path, quality_format: str, read_getter: Callable[[str], SequenceRead]) -> Iterator[SequenceReadPairwiseAlignment]:
        aligner = self.get_pe_forward_aligner()
        sam_file = self.perform_alignment(path, ReadType.PAIRED_END_1, quality_format, aligner)
        yield from self.parse_alignments(sam_file, read_getter)

    def align_pe_reverse(self, path: Path, quality_format: str, read_getter: Callable[[str], SequenceRead]) -> Iterator[SequenceReadPairwiseAlignment]:
        aligner = self.get_pe_reverse_aligner()
        sam_file = self.perform_alignment(path, ReadType.PAIRED_END_2, quality_format, aligner)
        yield from self.parse_alignments(sam_file, read_getter)

    def alignments_by_source(self, read_src: SampleReadSource, read_getter: Callable[[str], SequenceRead]) -> Iterator[SequenceReadPairwiseAlignment]:
        if isinstance(read_src, SampleReadSourceSingle):
            if read_src.read_type == ReadType.SINGLE_END:
                yield from self.align_single_end(read_src.path, read_src.quality_format, read_getter)
            elif read_src.read_type == ReadType.PAIRED_END_1:
                yield from self.align_pe_forward(read_src.path, read_src.quality_format, read_getter)
            elif read_src.read_type == ReadType.PAIRED_END_2:
                yield from self.align_pe_reverse(read_src.path, read_src.quality_format, read_getter)
            else:
                raise ValueError("Don't know how to handle a source with singular file of type `{}`".format(read_src.read_type))
        elif isinstance(read_src, SampleReadSourcePaired):
            yield from self.align_pe_forward(read_src.path_fwd, read_src.quality_format, read_getter)
            yield from self.align_pe_reverse(read_src.path_rev, read_src.quality_format, read_getter)


    def parse_alignments(self, sam_file: SamFile, read_getter: Callable[[str], SequenceRead]) -> Iterator[SequenceReadPairwiseAlignment]:
        yield from parse_alignments(
            sam_file, self.db,
            read_getter=read_getter,
            reattach_clipped_bases=True,
            min_hit_ratio=0.50,
            min_frag_len=15,
            print_tqdm_progressbar=self.print_tqdm
        )

    def perform_alignment(
            self,
            query_path: Path,
            read_type: ReadType,
            quality_format: str,
            aligner: AbstractPairwiseAligner
    ) -> SamFile:
        # ====== Files relative to cache dir.
        cache_relative_path = Path("alignments") / "{}.sam".format(query_path.stem)
        absolute_path = self.cache.cache_dir / cache_relative_path

        # ====== function bindings to pass to ComputationCache.
        def _call_aligner():
            absolute_path.parent.mkdir(exist_ok=True, parents=True)
            tmp_path = absolute_path.with_suffix('.sam.PARTIAL')
            aligner.align(
                query_path=query_path,
                output_path=tmp_path,
                read_type=read_type
            )
            shutil.move(src=tmp_path, dst=absolute_path)
            return SamFile(absolute_path, quality_format)

        # ====== Run the cached computation.
        return self.cache.call(
            fn=_call_aligner,
            relative_filepath=cache_relative_path,
            save=lambda path, obj: None,
            load=lambda path: SamFile(path, quality_format)
        )
