from contextlib import contextmanager
from pathlib import Path
from typing import List, Iterator, Union

import pysam
import numpy as np

from chronostrain.util.quality import ascii_to_phred
from .cigar import CigarElement, pysam_ordering
from chronostrain.util.sequences import Sequence, AllocatedSequence, UnknownNucleotideError

from chronostrain.logging import create_logger
from .util import *
logger = create_logger(__name__)


class SamLine:
    def __init__(self,
                 lineno: int,
                 raw_aln: pysam.AlignedSegment,
                 readname: str,
                 read_seq: Sequence,
                 read_phred: np.ndarray,
                 is_mapped: bool,
                 is_reverse_complemented: bool,
                 contig_name: str,
                 contig_map_idx: int,
                 cigar: List[CigarElement],
                 mate_pair: str,
                 mate_pos: str,
                 template_len: str
                 ):
        """
        Parse the line using the provided reference.
        """
        self.lineno = lineno
        self.raw_aln = raw_aln

        self.readname = readname
        self.read_seq: Sequence = read_seq
        self.read_phred = read_phred

        self.is_mapped = is_mapped
        self.is_reverse_complemented = is_reverse_complemented
        self.contig_name = contig_name
        self.contig_map_idx = contig_map_idx
        self.cigar = cigar

        self.mate_pair = mate_pair
        self.mate_pos = mate_pos
        self.template_len = template_len

    @property
    def read_len(self) -> int:
        return len(self.read_seq)

    def __str__(self):
        return "SamLine(L={lineno}):{tokens}".format(
            lineno=self.lineno,
            tokens=self.raw_aln.to_string()
        )

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def parse(lineno: int,
              aln_segment: pysam.AlignedSegment,
              prev_sam_line: Union['SamLine', None],
              quality_format: str) -> 'SamLine':
        readname = aln_segment.query_name
        map_flag = aln_segment.flag
        is_secondary_alignment = has_sam_flag(map_flag, SamFlags.SecondaryAlignment)

        if is_secondary_alignment:
            if prev_sam_line is None:
                raise RuntimeError(
                    "Unexpected SAM file output. Line ({lineno}) was flagged as a secondary alignment, "
                    "but this was the first line to be parsed.".format(
                        lineno=lineno
                    )
                )
            elif readname != prev_sam_line.readname:
                raise RuntimeError(
                    "Unexpected SAM file output. Line ({lineno}) was flagged as a secondary alignment, "
                    "but the previous line described a different input read ID.".format(
                        lineno=lineno
                    )
                )

            read_seq = prev_sam_line.read_seq
            read_phred = prev_sam_line.read_phred
        else:
            read_seq = AllocatedSequence(aln_segment.query_alignment_sequence)
            read_phred = ascii_pysam_to_phred(aln_segment.query_alignment_qualities, quality_format)

        cigar = [
            CigarElement(pysam_ordering[op], n)
            for op, n in aln_segment.cigartuples
        ]

        return SamLine(
            lineno=lineno,
            raw_aln=aln_segment,
            readname=readname,
            read_seq=read_seq,
            read_phred=read_phred,
            is_mapped=not has_sam_flag(map_flag, SamFlags.SegmentUnmapped),
            is_reverse_complemented=has_sam_flag(map_flag, SamFlags.SeqReverseComplement),
            contig_name=aln_segment.reference_name,
            contig_map_idx=aln_segment.reference_start,
            cigar=cigar,
            mate_pair=aln_segment.next_reference_id,
            mate_pos=aln_segment.next_reference_start,
            template_len=aln_segment.template_length
        )


def num_mismatches_from_xm(match_tag: str):
    head = "XM:i:"
    if not match_tag.startswith(head):
        raise RuntimeError("Expected XM tag to start with `{}`, got `{}`.".format(
            head, match_tag
        ))
    num_mismatches = int(match_tag[len(head):])
    return num_mismatches


@contextmanager
def open_with_pysam(file_path: Path):
    if file_path.suffix == ".sam":
        f = pysam.AlignmentFile(file_path, "r")
    elif file_path.suffix == ".bam":
        f = pysam.AlignmentFile(file_path, "rb")
    else:
        raise RuntimeError("Unrecognized output suffix {}".format(file_path.suffix))
    yield f
    f.close()


class SamIterator:
    def __init__(self, file_path: Path, quality_format: str):
        """
        :param file_path:
        :param quality_format: An option (as documented in Bio.SeqIO.QualityIO) for the quality score format.
        """
        self.file_path = file_path
        self.quality_format = quality_format

    def num_lines(self) -> int:
        with open_with_pysam(self.file_path) as f:
            return sum(1 for _ in f)

    def mapped_lines(self) -> Iterator[SamLine]:
        prev_sam_line: Union[SamLine, None] = None
        with open_with_pysam(self.file_path) as f:
            for line_idx, aln_segment in enumerate(f):
                try:
                    sam_line = self.parse_line(aln_segment, line_idx, prev_sam_line)
                except UnknownNucleotideError as e:
                    raise RuntimeError(
                        f"Encountered unknown nucleotide {e.nucleotide} "
                        f"while reading {self.file_path}, line {line_idx+1}"
                    )
                if sam_line.is_mapped:
                    yield sam_line

                prev_sam_line = sam_line

    def parse_line(self, aln_segment: pysam.AlignedSegment, line_idx: int, prev_line: SamLine) -> SamLine:
        return SamLine.parse(
            lineno=line_idx+1,
            aln_segment=aln_segment,
            prev_sam_line=prev_line,
            quality_format=self.quality_format
        )
