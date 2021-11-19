"""
Provides interfaces for alignments to be used throughout chronostrain module.
Each implementation wraps around the more general function bindings from chronostrain.util.external (which are
simple command-line interfaces).
"""

from abc import abstractmethod
from pathlib import Path
from typing import Optional

from chronostrain.util.external import *

from chronostrain.config import create_logger
logger = create_logger(__name__)


class AbstractPairwiseAligner(object):
    @abstractmethod
    def align(self, query_path: Path, output_path: Path):
        pass


class BwaAligner(AbstractPairwiseAligner):
    def __init__(self,
                 reference_path: Path,
                 min_seed_len: int,
                 num_threads: int,
                 report_all_alignments: bool):
        self.reference_path = reference_path
        self.min_seed_len = min_seed_len
        self.num_threads = num_threads
        self.report_all_alignments = report_all_alignments
        bwa_index(self.reference_path)

    def align(self, query_path: Path, output_path: Path):
        bwa_mem(
            output_path=output_path,
            reference_path=self.reference_path,
            read_path=query_path,
            min_seed_length=self.min_seed_len,
            num_threads=self.num_threads,
            report_all_alignments=self.report_all_alignments,
            match_score=1,
            mismatch_penalty=1,
            off_diag_dropoff=75,
        )


class BowtieAligner(AbstractPairwiseAligner):
    def __init__(self,
                 reference_path: Path,
                 index_basepath: Path,
                 index_basename: str,
                 num_threads: int,
                 num_report_alignments: Optional[int] = None):
        self.index_basepath = index_basepath
        self.index_basename = index_basename
        self.num_report_alignments = num_report_alignments
        self.num_threads = num_threads
        self.index_trace_path = self.index_basepath / f"{index_basename}.bt2trace"

        self.quality_format = 'phred33'
        logger.warn("Bowtie2 being initialized using default setting `phred33`. "
                    "(TODO: implement a universal definition across Biopython and bowtie2.)")

        if not self.index_trace_path.exists():  # only create if this hasn't been run yet.
            bowtie2_build(
                refs_in=[reference_path],
                index_basepath=self.index_basepath,
                index_basename=self.index_basename,
                quiet=True,
                n_threads=num_threads
            )
            self.index_trace_path.touch(exist_ok=True)  # Create an empty file to indicate that this finished.

    def align(self, query_path: Path, output_path: Path):
        bowtie2(
            index_basepath=self.index_basepath,
            index_basename=self.index_basename,
            unpaired_reads=query_path,
            out_path=output_path,
            quality_format=self.quality_format,
            report_k_alignments=self.num_report_alignments,
            num_threads=self.num_threads,
            aln_seed_num_mismatches=1,
            aln_seed_len=6,
            aln_seed_interval_fn=bt2_func_constant(3),
            aln_gbar=1,
            aln_n_ceil=bt2_func_linear(0, .1),
            score_mismatch_penalty=(1, 0),
            score_min_fn=bt2_func_linear(0, -0.1),
            score_read_gap_penalty=(5, 1),
            score_ref_gap_penalty=(5, 1),
            local=False,
            sam_suppress_noalign=True
        )
