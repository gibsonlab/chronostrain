from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.alignments.multiple import MarkerMultipleFragmentAlignment
from chronostrain.util.cache import ComputationCache
from chronostrain.util.external import run_glopp
from chronostrain.util.sequences import NucleotideDtype, nucleotide_GAP_z4
from ..cache import ReadsComputationCache
from .classes import MarkerContig

from chronostrain.config import create_logger
from .preprocess import _z4_base_ordering, to_bam, to_vcf
logger = create_logger(__name__)


class CachedGloppVariantAssembly(object):
    def __init__(self,
                 reads: TimeSeriesReads,
                 alignment: MarkerMultipleFragmentAlignment,
                 quality_lower_bound: float = 20,
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

        self.relative_dir = Path(f"glopp/{self.alignment.marker.id}")
        self.absolute_dir = self.cache.cache_dir / self.relative_dir
        self.variant_counts = self.count_variants()

    def count_variants(self) -> np.ndarray:
        """
        :return: An (N x 5) array of variant counts.
        """
        counts = np.zeros(shape=(self.alignment.num_bases(), len(_z4_base_ordering)), dtype=int)

        def add_to_tally(seq_aln_row, quality):
            # for each of A,C,G,T,- tally the variant, provided that the quality score is high enough.
            for b_idx, z4base in enumerate(_z4_base_ordering):
                z4_base_locs = np.where((seq_aln_row == z4base) & (quality > self.quality_lower_bound))[0]
                counts[first_idx + z4_base_locs, b_idx] += 1

        for read in self.alignment.reads(reverse=True):
            row_idx = self.alignment.get_index_of(read, True)
            row = self.alignment.read_multi_alignment[row_idx]
            first_idx, last_idx = self.alignment.aln_gapped_boundary_of_row(row_idx)

            row = row[first_idx:last_idx + 1]
            qual = np.zeros(shape=row.shape, dtype=float)
            qual[row != nucleotide_GAP_z4] = read.quality
            add_to_tally(row, qual)
        for read in self.alignment.reads(reverse=False):
            row_idx = self.alignment.get_index_of(read, False)
            row = self.alignment.read_multi_alignment[row_idx]
            first_idx, last_idx = self.alignment.aln_gapped_boundary_of_row(row_idx)

            row = row[first_idx:last_idx + 1]
            qual = np.zeros(shape=row.shape, dtype=float)
            qual[row != nucleotide_GAP_z4] = read.quality[::-1]
            add_to_tally(row, qual)
        return counts

    def prepare_glopp_input(self, ploidy: int) -> Tuple[Path, Path]:
        bam_path = self.prepare_bam(ploidy)
        vcf_path = self.prepare_vcf(self.variant_counts, ploidy)
        return bam_path, vcf_path

    def run_glopp(self, num_variants: int) -> List[MarkerContig]:
        subdir = f"ploidy_{num_variants}/output"
        phasing_rel_output_dir = self.relative_dir / subdir
        phasing_abs_output_dir = self.absolute_dir / subdir
        expected_rel_output_path = phasing_rel_output_dir / f"{self.alignment.marker.id}_phasing.txt"
        expected_abs_output_path = phasing_abs_output_dir / f"{self.alignment.marker.id}_phasing.txt"

        def _call():
            bam, vcf = self.prepare_glopp_input(ploidy=num_variants)
            run_glopp(
                bam_path=bam,
                vcf_path=vcf,
                output_dir=phasing_abs_output_dir,
                ploidy=num_variants
            )
            return self.parse_marker_contigs(expected_abs_output_path, num_variants)

        return self.cache.call(
            relative_filepath=expected_rel_output_path,
            fn=_call,
            save=lambda p, o: None,
            load=lambda p: self.parse_marker_contigs(expected_abs_output_path, num_variants)
        )

    def parse_marker_contigs(self, path: Path, expected_ploidy: int) -> List[MarkerContig]:
        """
        :param path: The glopp output to parse.
        :return: A length-C array of MarkerContig instances, where the c-th instance represents the N-ploidy
        "haplotype" assembly of the c-th region.
        """
        logger.debug(f"Parsing Marker assembly from {str(path)}")
        marker = self.alignment.marker
        all_positions: List[List[int]] = []
        contig_objects: List[MarkerContig] = []

        # First pass through file: verify file format and grab all positions.
        with open(path, "r") as glopp_file:
            glopp_contig_name = next(glopp_file).strip()[2:-2]
            if glopp_contig_name != marker.id:
                raise ValueError(
                    "Glopp output contained a different contig ID ({}) than what was expected ({}).".format(
                        glopp_contig_name, marker.id
                    )
                )

            cur_positions: List[int] = []
            for hap_line in glopp_file:
                if hap_line.startswith("-") or hap_line.startswith("*"):
                    all_positions.append(cur_positions)
                    cur_positions = []
                else:
                    tokens = hap_line.strip().split("\t")
                    if len(tokens) != (1 + 2 * expected_ploidy):
                        raise ValueError("Expected glopp output to have 2k+1 = {} columns, but got {}".format(
                            1 + 2 * expected_ploidy,
                            len(tokens)
                        ))

                    pos_order, pos = (int(token) for token in tokens[0].split(":"))
                    cur_positions.append(pos)

        # Second pass: Parse assembly.
        with open(path, "r") as glopp_file:
            # pass the header line.
            next(glopp_file)

            # parse.
            for contig_pos in all_positions:
                contig_assembly = np.empty(shape=(len(contig_pos), expected_ploidy), dtype=NucleotideDtype)
                for pos_idx, pos in enumerate(contig_pos):
                    hap_line = next(glopp_file)
                    tokens = hap_line.strip().split("\t")
                    for k in range(expected_ploidy):
                        contig_assembly[pos_idx, k] = int(tokens[k + 1])
                intermediate_line = next(glopp_file)
                assert intermediate_line.startswith("--") or intermediate_line.startswith("**")
                contig_objects.append(MarkerContig(marker, contig_pos, contig_assembly))
        return contig_objects

    def prepare_bam(self, ploidy: int) -> Path:
        target_path = self.absolute_dir / f"ploidy_{ploidy}" / "alignments.bam"
        target_path.parent.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Creating BAM file {str(target_path)}.")
        to_bam(self.alignment, target_path)
        return target_path

    def prepare_vcf(self, variants: np.array, ploidy: int) -> Path:
        target_path = self.absolute_dir / f"ploidy_{ploidy}" / "variants.vcf"
        target_path.parent.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Creating VCF file {str(target_path)}.")
        to_vcf(self.alignment, variants, ploidy, target_path)
        return target_path
