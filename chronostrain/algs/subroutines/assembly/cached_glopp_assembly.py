from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.cache import ComputationCache
from chronostrain.util.external import run_glopp
from chronostrain.util.flopp import FloppMarkerAssembly
from chronostrain.util.flopp.parser import FloppParser
from chronostrain.util.flopp.preprocess import z4_base_ordering, to_sam, to_vcf
from chronostrain.util.sequences import *
from ..cache import ReadsComputationCache

from chronostrain.config import create_logger
logger = create_logger(__name__)


class CachedGloppVariantAssembly(object):
    def __init__(self,
                 reads: TimeSeriesReads,
                 alignment: MarkerMultipleFragmentAlignment,
                 quality_lower_bound: float = 20,
                 variant_count_lower_bound: int = 5,
                 cache_override: Optional[ComputationCache] = None
                 ):
        """
        :param reads: The input time-series reads.
        :param alignment: The multiple alignment for a particular parker.
        :param quality_lower_bound: A lower bound for the quality score at which to (roughly) call variants.
        :param cache_override: If specified, uses this cache instead of the default cache depending on the reads.
        """
        self.reads = reads

        if cache_override is not None:
            self.cache = cache_override
        else:
            self.cache = ReadsComputationCache(reads)

        self.alignment: MarkerMultipleFragmentAlignment = alignment
        self.quality_lower_bound = quality_lower_bound
        self.variant_count_lower_bound = variant_count_lower_bound

        self.relative_dir = Path(f"glopp/{self.alignment.canonical_marker.id}")
        self.absolute_dir = self.cache.cache_dir / self.relative_dir

        self.sam_path = self.absolute_dir / "alignments.sam"
        self.vcf_path = self.absolute_dir / "variants.vcf"

    @property
    def num_times(self) -> int:
        return len(self.reads)

    def count_variants(self) -> np.ndarray:
        """
        :return: An (N x 5) array of variant counts.
        """
        counts = np.zeros(shape=(self.alignment.num_bases(), len(z4_base_ordering)), dtype=int)

        def add_to_tally(seq_aln_row, quality):
            # for each of A,C,G,T,- tally the variant, provided that the quality score is high enough.
            for b_idx, z4base in enumerate(z4_base_ordering):
                z4_base_locs = np.where((seq_aln_row == z4base) & (quality > self.quality_lower_bound))[0]
                counts[first_idx + z4_base_locs, b_idx] += 1

        for read in self.alignment.reads(revcomp=True):
            row = self.alignment.get_aligned_read_seq(read, True)
            first_idx, last_idx = self.alignment.aln_gapped_boundary(read, True)

            read_start_clip, read_end_clip = self.alignment.num_clipped_bases(read, True)
            _slice = slice(read_start_clip, len(read) - read_end_clip)

            row = row[first_idx:last_idx + 1]
            qual = np.zeros(shape=row.shape, dtype=float)
            qual[row != nucleotide_GAP_z4] = read.quality[_slice]
            add_to_tally(row, qual)
        for read in self.alignment.reads(revcomp=False):
            row = self.alignment.get_aligned_read_seq(read, False)
            first_idx, last_idx = self.alignment.aln_gapped_boundary(read, False)

            read_start_clip, read_end_clip = self.alignment.num_clipped_bases(read, False)
            _slice = slice(read_start_clip, len(read) - read_end_clip)

            row = row[first_idx:last_idx + 1]
            qual = np.zeros(shape=row.shape, dtype=float)
            qual[row != nucleotide_GAP_z4] = read.quality[::-1][_slice]
            add_to_tally(row, qual)

        return counts

    def prepare_glopp_input(self) -> Tuple[Path, Path]:
        sam_path = self.prepare_bam()
        vcf_path = self.prepare_vcf()
        return sam_path, vcf_path

    def run(self, num_variants: Optional[int] = None) -> FloppMarkerAssembly:
        if num_variants is None:
            subdir = "output"
        else:
            subdir = f"output_ploidy_{num_variants}"
        rel_output_dir = self.relative_dir / subdir
        abs_output_dir = self.absolute_dir / subdir
        phasing_rel_output_path = rel_output_dir / f"{self.alignment.canonical_marker.name}_phasing.txt"
        phasing_abs_output_path = abs_output_dir / f"{self.alignment.canonical_marker.name}_phasing.txt"
        partition_abs_output_path = abs_output_dir / f"{self.alignment.canonical_marker.name}_part.txt"

        def _call():
            self.prepare_glopp_input()

            # use_mec_score=True is better for learning metagenomic assemblies, since it
            # "removes" the assumption that the rel abundances are equal.
            run_glopp(
                sam_path=self.sam_path,
                vcf_path=self.vcf_path,
                output_dir=abs_output_dir,
                ploidy=num_variants,
                use_mec_score=True,
                allele_error_rate=0.0005
            )

            return FloppParser(self.vcf_path, self.alignment).parse(
                phasing_abs_output_path, partition_abs_output_path
            )

        return self.cache.call(
            relative_filepath=phasing_rel_output_path,
            fn=_call,
            save=lambda p, o: None,
            load=lambda p: FloppParser(self.vcf_path, self.alignment).parse(
                phasing_abs_output_path, partition_abs_output_path
            )
        )

    def prepare_bam(self):
        self.sam_path.parent.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Creating BAM file {str(self.sam_path)}.")
        to_sam(self.alignment.canonical_marker,
               self.alignment,
               self.sam_path)
        return self.sam_path

    def prepare_vcf(self):
        self.vcf_path.parent.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Creating VCF file {str(self.vcf_path)}.")
        to_vcf(self.alignment.canonical_marker,
               self.alignment,
               self.count_variants(),
               self.vcf_path,
               ploidy=None,
               variant_count_lower_bound=self.variant_count_lower_bound)
        return self.vcf_path
