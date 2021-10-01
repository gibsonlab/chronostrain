from typing import Tuple, Iterator

import numpy as np
import re

from chronostrain.database import StrainDatabase
from chronostrain.model import Marker
from chronostrain.util.sam_handler import SamHandler
from chronostrain.util.sequences import map_nucleotide_to_z4, nucleotides_to_z4, z4_to_nucleotides


class NoSuchMarkerException(BaseException):
    pass


class Pileup(object):
    """
    Represents a pileup of variant counts (not equal to reference base) observed in reads, for a particular marker.
    """
    def __init__(self, marker: Marker):
        self.marker = marker
        self.seq = nucleotides_to_z4(marker.seq)
        self.totals = np.zeros(shape=(len(marker.seq), 4), dtype=int)
        self.variant_to_reads = {}
        self.reads_to_variants = {}
        self.overall_total = 0

    def increment(self, position: int, base: str, quality: float):
        if self.marker.seq[position] == base:
            return
        self.totals[position, map_nucleotide_to_z4(base)] += quality
        self.overall_total += quality

    def clear(self, position: int, base: str):
        total = self.totals[position, map_nucleotide_to_z4(base)]
        self.totals[position, map_nucleotide_to_z4(base)] = 0
        self.overall_total -= total

    def maximal_variant_combinations(self) -> Iterator[Tuple[str, np.ndarray]]:
        max_variants = np.argmax(self.totals, axis=1)
        new_seq = self.seq.copy()
        counts = self.totals[np.arange(len(self.totals)), max_variants]

        selector = counts > 0  # TODO: try a subset of this instead. Should iterate through all combinations.
        new_seq[selector] = max_variants[selector]
        yield z4_to_nucleotides(new_seq), selector

    def subtract_variant(self, variant_seq: str):
        for pos, char in enumerate(variant_seq):
            self.clear(pos, char)


class MarkerPileups(object):
    def __init__(self, db: StrainDatabase):
        self.pileups = {
            marker.id: Pileup(marker)
            for marker in db.all_markers()
        }
        self.db = db

    def marker_with_largest_pileup(self) -> Marker:
        highest_total = -1
        target_marker_name = None
        for marker_name, pileup in self.pileups.items():
            if pileup.overall_total > highest_total:
                target_marker_name = marker_name
        if target_marker_name is None:
            raise NoSuchMarkerException()
        return self.db.get_marker(target_marker_name)

    def proposal_variants(self, marker: Marker) -> Iterator[Tuple[str, np.ndarray]]:
        pileup = self.pileups[marker.id]
        yield from pileup.maximal_variant_combinations()

    def accept_variant(self, marker: Marker, variant_seq: str):
        self.pileups[marker.id].subtract_variant(variant_seq)

    @staticmethod
    def find_start_clip(cigar_tag):
        split_cigar = re.findall('\d+|\D+', cigar_tag)
        if split_cigar[1] == 'S':
            return int(split_cigar[0])
        return 0

    def add_aligned_evidence(self, sam_output: SamHandler):
        """
        Keep a pileup of aligned variants.
        :param sam_output:
        :return:
        """
        for samline in sam_output.mapped_lines():
            accession_token, name_token, id_token = samline.contig_name.split("|")
            mapped_marker = self.db.get_marker(
                # Assumes that the reference marker was stored automatically using Marker.to_seqrecord().
                id_token
            )

            # Other attributes readable from SAM format.
            read_quality = samline.phred_quality
            read_seq = samline.read
            read_len = len(samline.read)
            ref_index = int(samline.map_pos_str) - self.find_start_clip(samline.cigar) - 1

            # Loop through the read's nucleotides, one by one.
            read_index = 0
            if ref_index < 0:
                read_index += -ref_index
                ref_index = 0

            pileup = self.pileups[mapped_marker.id]
            while ref_index < len(mapped_marker) and read_index < read_len:
                pileup.increment(ref_index, read_seq[read_index], read_quality[read_index])
                ref_index += 1
                read_index += 1
