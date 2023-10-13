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

from chronostrain.logging import create_logger
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
                 mem_discard_threshold: int,
                 chain_drop_threshold: float,
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
        self.mem_discard_threshold = mem_discard_threshold
        self.chain_drop_threshold = chain_drop_threshold
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

    def post_process_attach_id_suffix(self, sam_path: Path, id_suffix: str):
        output_path = sam_path.parent / f'{sam_path.name}.processed'
        with open(sam_path, 'r') as in_f, open(output_path, 'w') as out_f:
            # only keep mapped reads.
            for line in in_f:
                if line.startswith("@"):
                    out_f.write(line)
                else:
                    tokens = line.rstrip().split('\t')

                    # BWA-MEM idiosyncracy: aligner removes the paired-end identifiers '/1', '/2'.
                    read_id = tokens[0]
                    tokens[0] = f'{read_id}{id_suffix}'
                    print('\t'.join(tokens), file=out_f)
        sam_path.unlink()
        output_path.rename(sam_path)

    def align(self, query_path: Path, output_path: Path, read_type: ReadType, exclude_unmapped: bool = False):
        bwa_mem(
            output_path=output_path,
            reference_path=self.reference_path,
            read_path=query_path,
            min_seed_length=self.min_seed_len,
            reseed_ratio=self.reseed_ratio,
            mem_discard_threshold=self.mem_discard_threshold,
            chain_drop_threshold=self.chain_drop_threshold,
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
            bwa_cmd=self.bwa_command,
        )

        if read_type == ReadType.PAIRED_END_1:
            self.post_process_attach_id_suffix(output_path, '/1')
        elif read_type == ReadType.PAIRED_END_2:
            self.post_process_attach_id_suffix(output_path, '/2')


logger.info("If invoked, bowtie2 will initialize using default setting `phred33`. "
               "(TODO: implement some flexibility here.)")


class BowtieAligner(AbstractPairwiseAligner):
    def __init__(self,
                 reference_path: Path,
                 index_basepath: Path,
                 index_basename: str,
                 num_threads: int,
                 seed_length: int,
                 seed_extend_failures: int,
                 num_reseeds: int,
                 score_min_fn: str,
                 score_match_bonus: int,
                 score_mismatch_penalty: Tuple[int, int],
                 score_read_gap_penalty: Tuple[int, int],
                 score_ref_gap_penalty: Tuple[int, int],
                 index_offrate: int = 1,
                 index_ftabchars: int = 13,
                 align_offrate: Optional[int] = None,
                 seed_num_mismatches: int = 0,
                 num_report_alignments: Optional[int] = None,
                 report_all_alignments: bool = False):
        self.index_basepath = index_basepath
        self.index_basename = index_basename
        self.index_offrate = index_offrate
        self.index_ftabchars = index_ftabchars
        self.align_offrate = align_offrate
        self.num_report_alignments = num_report_alignments
        self.report_all_alignments = report_all_alignments
        self.num_threads = num_threads

        # Alignment params
        self.seed_length = seed_length
        self.seed_num_mistmatches = seed_num_mismatches
        self.seed_extend_failures = seed_extend_failures
        self.num_reseeds = num_reseeds
        self.score_min_fn = score_min_fn
        self.score_match_bonus = score_match_bonus
        self.score_mismatch_penalty = score_mismatch_penalty
        self.score_read_gap_penalty = score_read_gap_penalty
        self.score_ref_gap_penalty = score_ref_gap_penalty

        self.index_trace_path = self.index_basepath / f"{index_basename}.bt2_trace"
        self.quality_format = 'phred33'

        if not self.index_trace_path.exists():  # only create if this hasn't been run yet.
            bowtie2_build(
                refs_in=[reference_path],
                index_basepath=self.index_basepath,
                index_basename=self.index_basename,
                offrate=self.index_offrate,  # default is 5; but we want to optimize for the -a option.
                ftabchars=self.index_ftabchars,
                quiet=True,
                threads=self.num_threads,
            )
            self.index_trace_path.touch(exist_ok=True)  # Create an empty file to indicate that this finished.
        else:
            logger.debug("pre-built bowtie2 index found at {}".format(
                str(self.index_trace_path)
            ))

    def align(self, query_path: Path, output_path: Path, read_type: ReadType):
        # return self.align_end_to_end(query_path, output_path)
        self.align_local(query_path, output_path)
        # self.post_process_sam(output_path)

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
            aln_seed_num_mismatches=0,  # -N
            aln_seed_len=self.seed_length,  # -L
            aln_seed_interval_fn=bt2_func_constant(7),
            aln_gbar=1,
            offrate=self.align_offrate,
            effort_seed_ext_failures=self.seed_extend_failures,  # -D
            local=False,
            effort_num_reseeds=self.num_reseeds,  # -R
            score_match_bonus=self.score_match_bonus,
            score_min_fn=self.score_min_fn,
            score_mismatch_penalty=self.score_mismatch_penalty,
            score_read_gap_penalty=self.score_read_gap_penalty,
            score_ref_gap_penalty=self.score_ref_gap_penalty,
            sam_suppress_noalign=True
        )

    def align_local(self, query_path: Path, output_path: Path):
        # This implements the --very-sensitive-local setting with more extensive seeding.
        #-D 20 -R 3 -N 0 -L 20 -i S,1,0.50
        bowtie2(
            index_basepath=self.index_basepath,
            index_basename=self.index_basename,
            unpaired_reads=query_path,
            out_path=output_path,
            quality_format=self.quality_format,
            report_k_alignments=self.num_report_alignments,
            report_all_alignments=self.report_all_alignments,
            num_threads=self.num_threads,
            aln_seed_num_mismatches=0,  # -N
            aln_seed_len=self.seed_length,  # -L
            aln_seed_interval_fn=bt2_func_constant(7),
            aln_gbar=1,
            offrate=self.align_offrate,
            effort_seed_ext_failures=self.seed_extend_failures,  # -D
            local=True,
            effort_num_reseeds=self.num_reseeds,  # -R
            score_min_fn=self.score_min_fn,
            score_match_bonus=self.score_match_bonus,
            score_mismatch_penalty=self.score_mismatch_penalty,
            score_read_gap_penalty=self.score_read_gap_penalty,
            score_ref_gap_penalty=self.score_ref_gap_penalty,
            sam_suppress_noalign=True
        )
