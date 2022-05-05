"""
Provides interfaces for alignments to be used throughout chronostrain module.
Each implementation wraps around the more general function bindings from chronostrain.util.external (which are
simple command-line interfaces).
"""

from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

from chronostrain.model.io import ReadType
from chronostrain.util.external import *
from chronostrain.util.alignments.sam.sam_iterators import *

from chronostrain.config import create_logger
logger = create_logger(__name__)


class AbstractPairwiseAligner(object):
    @abstractmethod
    def align(self, query_path: Path, output_path: Path, read_type: ReadType):
        pass


class SmithWatermanAligner(AbstractPairwiseAligner):
    def __init__(self,
                 reference_path: Path,
                 match_score: int,
                 mismatch_penalty: int,
                 gap_open_penalty: int,
                 gap_extend_penalty: int,
                 score_threshold: int):
        self.reference_path = reference_path
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_open_penalty = gap_open_penalty
        self.gap_extend_penalty = gap_extend_penalty
        self.score_threshold = score_threshold

    def align(self, query_path: Path, output_path: Path, read_type: ReadType):
        ssw_align(
            target_path=self.reference_path,
            query_path=query_path,
            match_score=self.match_score,
            mismatch_penalty=self.mismatch_penalty,
            gap_open_penalty=self.gap_open_penalty,
            gap_extend_penalty=self.gap_extend_penalty,
            output_path=output_path,
            best_of_strands=True,
            score_threshold=self.score_threshold
        )


class BwaAligner(AbstractPairwiseAligner):
    def __init__(self,
                 reference_path: Path,
                 min_seed_len: int,
                 reseed_ratio: float,
                 bandwidth: int,
                 num_threads: int,
                 report_all_alignments: bool,
                 match_score: int,
                 mismatch_penalty: int,
                 off_diag_dropoff: int,
                 gap_open_penalty: Union[int, Tuple[int, int]],
                 gap_extend_penalty: Union[int, Tuple[int, int]],
                 clip_penalty: int,
                 score_threshold: int,
                 bwa_command: str = 'bwa-mem2'):
        self.reference_path = reference_path
        self.min_seed_len = min_seed_len
        self.reseed_ratio = reseed_ratio
        self.bandwidth = bandwidth
        self.num_threads = num_threads
        self.report_all_alignments = report_all_alignments
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.off_diag_dropoff = off_diag_dropoff
        self.gap_open_penalty = gap_open_penalty
        self.gap_extend_penalty = gap_extend_penalty
        self.clip_penalty = clip_penalty
        self.score_threshold = score_threshold
        self.bwa_command = bwa_command

        self.index_trace_path = self.reference_path.with_suffix(f'.{bwa_command}_trace')

        if not self.index_trace_path.exists():  # only create if this hasn't been run yet.
            bwa_index(self.reference_path, bwa_cmd=bwa_command)
            self.index_trace_path.touch(exist_ok=True)  # Create an empty file to indicate that this finished.
        else:
            logger.debug("pre-built bwa index found at {}".format(
                str(self.index_trace_path)
            ))

    def post_process(self, sam_path: Path, output_path: Path, id_suffix: str):
        with open(sam_path, 'r') as in_f, open(output_path, 'w') as out_f:
            # only keep mapped reads.
            for line in cull_repetitive_templates(mapped_only(skip_headers(in_f))):
                tokens = line.rstrip().split('\t')

                # BWA-MEM idiosyncracy: aligner removes the paired-end identifiers '/1', '/2'.
                read_id = tokens[0]
                tokens[0] = f'{read_id}{id_suffix}'
                print('\t'.join(tokens), file=out_f)

    def align(self, query_path: Path, output_path: Path, read_type: ReadType):
        tmp_sam = output_path.parent / (f'{output_path.stem}_bwa.sam')

        bwa_mem(
            output_path=tmp_sam,
            reference_path=self.reference_path,
            read_path=query_path,
            min_seed_length=self.min_seed_len,
            reseed_ratio=self.reseed_ratio,
            bandwidth=self.bandwidth,
            num_threads=self.num_threads,
            report_all_alignments=self.report_all_alignments,
            match_score=self.match_score,
            mismatch_penalty=self.mismatch_penalty,
            off_diag_dropoff=self.off_diag_dropoff,
            gap_open_penalty=self.gap_open_penalty,
            gap_extend_penalty=self.gap_extend_penalty,
            clip_penalty=self.clip_penalty,
            unpaired_penalty=0,
            soft_clip_for_supplementary=True,
            score_threshold=self.score_threshold,
            bwa_cmd=self.bwa_command
        )

        if read_type == ReadType.PAIRED_END_1:
            self.post_process(tmp_sam, output_path, '/1')
            tmp_sam.unlink()
        elif read_type == ReadType.PAIRED_END_2:
            self.post_process(tmp_sam, output_path, '/2')
            tmp_sam.unlink()
        else:
            tmp_sam.rename(output_path)


logger.warn("If invoked, bowtie2 will initialize using default setting `phred33`. "
            "(TODO: implement a universal definition across Biopython and bowtie2.)")


class BowtieAligner(AbstractPairwiseAligner):
    def __init__(self,
                 reference_path: Path,
                 index_basepath: Path,
                 index_basename: str,
                 num_threads: int,
                 num_reseeds: int,
                 score_min_fn: str,
                 score_mismatch_penalty: Tuple[int, int],
                 score_read_gap_penalty: Tuple[int, int],
                 score_ref_gap_penalty: Tuple[int, int],
                 num_report_alignments: Optional[int] = None,
                 report_all_alignments: bool = False):
        self.index_basepath = index_basepath
        self.index_basename = index_basename
        self.num_report_alignments = num_report_alignments
        self.report_all_alignments = report_all_alignments
        self.num_threads = num_threads

        # Alignment params
        self.num_reseeds = num_reseeds
        self.score_min_fn = score_min_fn
        self.score_mismatch_penalty = score_mismatch_penalty
        self.score_read_gap_penalty = score_read_gap_penalty
        self.score_ref_gap_penalty = score_ref_gap_penalty

        self.index_trace_path = self.index_basepath / f"{index_basename}.bt2trace"

        self.quality_format = 'phred33'

        if not self.index_trace_path.exists():  # only create if this hasn't been run yet.
            bowtie2_build(
                refs_in=[reference_path],
                index_basepath=self.index_basepath,
                index_basename=self.index_basename,
                quiet=True,
                n_threads=num_threads
            )
            self.index_trace_path.touch(exist_ok=True)  # Create an empty file to indicate that this finished.
        else:
            logger.debug("pre-built bowtie2 index found at {}".format(
                str(self.index_trace_path)
            ))

    def align(self, query_path: Path, output_path: Path, read_type: ReadType):
        return self.align_end_to_end(query_path, output_path)
        # return self.align_local(query_path, output_path)

    def align_end_to_end(self, query_path: Path, output_path: Path):
        bowtie2(
            index_basepath=self.index_basepath,
            index_basename=self.index_basename,
            unpaired_reads=query_path,
            out_path=output_path,
            quality_format=self.quality_format,
            report_k_alignments=self.num_report_alignments,
            report_all_alignments=self.report_all_alignments,
            num_threads=self.num_threads,
            aln_seed_num_mismatches=0,
            aln_seed_len=20,  # -L 20
            aln_seed_interval_fn=bt2_func_constant(7),
            aln_gbar=1,
            effort_seed_ext_failures=30,  # -D 30
            local=False,
            effort_num_reseeds=self.num_reseeds,
            score_min_fn=self.score_min_fn,
            score_mismatch_penalty=self.score_mismatch_penalty,
            score_read_gap_penalty=self.score_read_gap_penalty,
            score_ref_gap_penalty=self.score_ref_gap_penalty,
            sam_suppress_noalign=True
        )

    # def align_local(self, query_path: Path, output_path: Path):
    #     # This implements the --very-sensitive-local setting with more extensive seeding.
    #     #-D 20 -R 3 -N 0 -L 20 -i S,1,0.50
    #     bowtie2(
    #         index_basepath=self.index_basepath,
    #         index_basename=self.index_basename,
    #         unpaired_reads=query_path,
    #         out_path=output_path,
    #         quality_format=self.quality_format,
    #         report_k_alignments=self.num_report_alignments,
    #         report_all_alignments=self.report_all_alignments,
    #         num_threads=self.num_threads,
    #         aln_seed_num_mismatches=0,
    #         aln_seed_len=20,  # -L 20
    #         aln_seed_interval_fn=bt2_func_constant(7),
    #         aln_gbar=1,
    #         effort_seed_ext_failures=30,  # -D 30
    #         local=True,
    #         effort_num_reseeds=self.num_reseeds,  # -R 3
    #         score_min_fn=self.score_min_fn,
    #         score_match_bonus=self.score_match_bonus,
    #         score_mismatch_penalty=self.score_mismatch_penalty,
    #         score_read_gap_penalty=self.score_read_gap_penalty,
    #         score_ref_gap_penalty=self.score_ref_gap_penalty,
    #         sam_suppress_noalign=True
    #     )
