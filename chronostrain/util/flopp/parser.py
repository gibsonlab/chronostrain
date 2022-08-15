from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from .base import map_nucleotide_to_z4_with_special
from .classes import FloppMarkerContig, FloppMarkerAssembly
from ..alignments.multiple import MarkerMultipleFragmentAlignment
from ..sequences import NucleotideDtype

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class FloppParser(object):
    def __init__(self, vcf_path: Path, alignment: MarkerMultipleFragmentAlignment):
        self.vcf_path = vcf_path
        self.alignment = alignment

    def variant_allele_parser(self) -> Callable[[int, int], int]:
        """
        :return: A function representing the mapping (position, allele) -> (nucleotide).
        The undeterminable case "allele = -1" is by default treated as the reference base.
        """
        mappings: List[Dict[int, int]] = []
        with open(self.vcf_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue

                tokens = line.strip().split("\t")
                ref_allele = map_nucleotide_to_z4_with_special(tokens[3].strip())
                variants = [map_nucleotide_to_z4_with_special(x) for x in tokens[4].strip().split(",")]
                allele_to_nucleotide: Dict[int, int] = dict()
                allele_to_nucleotide[0] = ref_allele
                allele_to_nucleotide[-1] = ref_allele  # By default, map "-1" to reference base.
                for i, nucleotide in enumerate(variants):
                    allele_to_nucleotide[i+1] = nucleotide
                mappings.append(allele_to_nucleotide)

        def _apply(pos_idx: int, allele: int) -> int:
            return mappings[pos_idx][allele]

        return _apply

    def parse(self,
              glopp_phasing_path: Path,
              glopp_partition_path: Path) -> FloppMarkerAssembly:
        """
        :param glopp_phasing_path: The glopp phasing output file to parse.
        :param glopp_partition_path: The glopp partition output file to parse.
        :return: A length-C array of MarkerContig instances, where the c-th instance represents the N-ploidy
        "haplotype" assembly of the c-th region.
        """
        logger.debug(f"Parsing Marker assembly from {str(glopp_phasing_path)}")
        all_positions: List[int] = []  # A list of the variant positions, parsed from the phasing file.

        contig_positions_arr: List[np.ndarray] = []  # contig-indexed list of contig-specific positions.
        contig_assemblies: List[np.ndarray] = []  # contig-indexed list of assembled sequences.
        contig_read_counts_arr: List[np.ndarray] = []  # contig-indexed list of read counts per strand.

        parsed_ploidy: int = -1
        allele_parser = self.variant_allele_parser()

        # First pass through file: verify file format and grab all positions.
        with open(glopp_phasing_path, "r") as phasing_file:
            glopp_contig_name = next(phasing_file).strip()[2:-2]
            if glopp_contig_name != self.alignment.canonical_marker.name:
                raise ValueError(
                    "Glopp output contained a different contig ID ({}) than what was expected ({}).".format(
                        glopp_contig_name, self.alignment.canonical_marker.name
                    )
                )

            cur_positions: List[int] = []
            for hap_line in phasing_file:
                if hap_line.startswith("-") or hap_line.startswith("*"):
                    contig_positions_arr.append(np.array(cur_positions))
                    cur_positions = []
                else:
                    tokens = hap_line.strip().split("\t")
                    if (len(tokens) - 1) % 2 != 0:
                        raise ValueError("Expected glopp output to have 2k+1 columns, but got {}".format(
                            len(tokens)
                        ))
                    if parsed_ploidy < 0:
                        parsed_ploidy = (len(tokens) - 1) // 2
                        logger.debug(f"Found variant assembly with ploidy {parsed_ploidy}")

                    pos_order, pos = (int(token) for token in tokens[0].split(":"))
                    cur_positions.append(pos - 1)  # pos is 1-indexed.
                    all_positions.append(pos - 1)

        # Second pass: Parse assembly.
        with open(glopp_phasing_path, "r") as phasing_file:
            # pass the header line.
            next(phasing_file)

            # parse.
            abs_pos_idx = -1
            for contig_idx, contig_pos in enumerate(contig_positions_arr):
                contig_assembly = np.empty(shape=(len(contig_pos), parsed_ploidy), dtype=NucleotideDtype)

                # TODO: num_times is no longer relevant (since our strategy for time-correlations of raw counts doesn't really work).
                read_counts = np.zeros(shape=(parsed_ploidy, self.num_times), dtype=int)
                for contig_pos_idx, pos in enumerate(contig_pos):
                    hap_line = next(phasing_file)
                    abs_pos_idx += 1
                    tokens = hap_line.strip().split("\t")
                    for strand_idx in range(parsed_ploidy):
                        contig_assembly[contig_pos_idx, strand_idx] = allele_parser(
                            abs_pos_idx, int(tokens[strand_idx + 1])
                        )
                intermediate_line = next(phasing_file)
                assert intermediate_line.startswith("--") or intermediate_line.startswith("**")
                contig_assemblies.append(contig_assembly)
                contig_read_counts_arr.append(read_counts)

        # Parse the partition file: get the per-timepoint, per-base read counts.
        with open(glopp_partition_path, "r") as partition_file:
            strand_idx: int = -1
            for line in partition_file:
                if line.startswith("#"):
                    # Determine which partition we're in.
                    strand_idx = int(line.strip()[1:])
                else:
                    read_id_tok, first_pos_tok, last_pos_tok = line.strip().split("\t")

                    # Parse the read identifier.
                    t_tok, revcomp_tok, read_id = read_id_tok.split(';')
                    assert t_tok.startswith("T")
                    assert revcomp_tok.startswith("R")
                    t_idx = int(t_tok[1:])
                    revcomp = int(revcomp_tok[1:]) == 1

                    # Obtain the read's aligned sequence.
                    try:
                        read_obj = self.reads[t_idx].get_read(read_id)
                    except KeyError:
                        logger.error("Missing key `{}` from timepoint index `{}`.".format(
                            read_id, t_idx
                        ))
                        raise
                    read_full_seq = self.alignment.get_aligned_read_seq(read_obj, revcomp)

                    # Parse the read's first/last positions.
                    read_first_pos = all_positions[int(first_pos_tok) - 1]
                    read_last_pos = all_positions[int(last_pos_tok) - 1]

                    # For each contig, determine whether it overlaps ("hits") the read.
                    for contig_positions, contig_assembly, read_counts \
                            in zip(contig_positions_arr, contig_assemblies, contig_read_counts_arr):
                        # "Hit" = there exists some position on the contig that is between the read's first/last pos.
                        read_variants = read_full_seq[contig_positions]
                        assembly_variants = contig_assembly[:, strand_idx]
                        hit_positions = np.where(
                            (read_first_pos <= contig_positions)
                            & (contig_positions <= read_last_pos)
                            & (read_variants == assembly_variants)
                        )[0]

                        # increment overall read count for the strand.
                        if len(hit_positions) > 0:
                            read_counts[strand_idx, t_idx] += 1
        return FloppMarkerAssembly(
            self.alignment,
            [
                FloppMarkerContig(
                    canonical_marker=self.alignment.canonical_marker,
                    contig_idx=c_idx,
                    positions=contig_pos,
                    assembly=contig_assembly,
                    read_counts=read_counts
                )
                for c_idx, (contig_pos, contig_assembly, read_counts) in enumerate(
                    zip(
                        contig_positions_arr,
                        contig_assemblies,
                        contig_read_counts_arr
                    )
                )
            ],
            np.array(all_positions)
        )
