import enum
from pathlib import Path
from typing import List, Iterator, Union, Optional

import numpy as np

from chronostrain.util.quality import ascii_to_phred
from .cigar import CigarElement, parse_cigar, CigarOp
from chronostrain.util.sequences import SeqType, nucleotides_to_z4, UnknownNucleotideError

from chronostrain.config import create_logger
logger = create_logger(__name__)


class _SamTags(enum.Enum):
    ReadName = 0
    MapFlag = 1
    ContigName = 2
    MapPos = 3
    MapQuality = 4
    Cigar = 5
    MatePair = 6
    MatePos = 7
    TemplateLen = 8
    Read = 9
    Quality = 10


class SamFlags(enum.Enum):
    """
    The mapping types given in the MapFlag tag. The actual tag is given bitwise,
    so the presence of these tags is found as:
    (Line[SamTags.MapFlag] & MapFlags.flag == MapFlags.flag)
    """
    QueryHasMultipleSegments = 1
    SegmentsProperlyAligned = 2
    SegmentUnmapped = 4
    NextSegmentUnmapped = 8
    SeqReverseComplement = 16
    SeqNextSegmentReverseComplement = 32
    FirstSegment = 64
    LastSegment = 128
    SecondaryAlignment = 256
    FilterNotPassed = 512
    PCRorOptionalDuplicate = 1024
    SupplementaryAlignment = 2048


def _check_bit_flag(x: int, pow2: int) -> bool:
    return (x & pow2) == pow2


class SamLine:
    def __init__(self,
                 lineno: int,
                 plaintext_line: str,
                 readname: str,
                 read_seq: SeqType,
                 read_phred: np.ndarray,
                 is_mapped: bool,
                 is_reverse_complemented: bool,
                 contig_name: str,
                 contig_map_idx: int,
                 cigar: List[CigarElement],
                 mate_pair: str,
                 mate_pos: str,
                 template_len: str,
                 num_aligned_bases: Optional[int],
                 num_mismatches: Optional[int]
                 ):
        """
        Parse the line using the provided reference.

        :param lineno: The line number in the .sam file corresponding to this instance.
        :param plaintext_line: The raw line read from the .sam file.
        """
        self.lineno = lineno
        self.line = plaintext_line

        self.readname = readname
        self.read_seq = read_seq
        self.read_phred = read_phred

        self.is_mapped = is_mapped
        self.is_reverse_complemented = is_reverse_complemented
        self.contig_name = contig_name
        self.contig_map_idx = contig_map_idx
        self.cigar = cigar

        self.mate_pair = mate_pair
        self.mate_pos = mate_pos
        self.template_len = template_len

        self.num_aligned_bases = num_aligned_bases
        self.num_mismatches = num_mismatches

    @property
    def read_len(self) -> int:
        return len(self.read_seq)

    def __str__(self):
        return "SamLine(L={lineno}):{tokens}".format(
            lineno=self.lineno,
            tokens=self.line
        )

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def parse(lineno: int,
              plaintext_line: str,
              prev_sam_line: Union['SamLine', None],
              quality_format: str) -> 'SamLine':
        tokens = plaintext_line.strip().split('\t')

        readname = tokens[_SamTags.ReadName.value]
        map_flag = int(tokens[_SamTags.MapFlag.value])
        is_secondary_alignment = _check_bit_flag(map_flag, SamFlags.SecondaryAlignment.value)

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
            read_seq = nucleotides_to_z4(tokens[_SamTags.Read.value])
            read_phred = ascii_to_phred(tokens[_SamTags.Quality.value], quality_format)

        cigar = parse_cigar(tokens[_SamTags.Cigar.value])

        num_aligned_bases: Union[float, None] = None
        num_mismatches: Union[float, None] = None
        for optional_tag in tokens[11:]:
            '''
            The MD tag stores information about which bases match to the reference and is necessary
            for determining percent identity
            '''
            # MD tag: shows percent identity
            if optional_tag.startswith("XM:"):
                num_aligned_bases = sum(
                    cigar_el.num
                    for cigar_el in cigar
                    if cigar_el.op == CigarOp.ALIGN or cigar_el.op == CigarOp.MATCH or cigar_el.op == CigarOp.MISMATCH
                )
                num_mismatches = num_mismatches_from_xm(optional_tag)
            else:
                pass

        return SamLine(
            lineno=lineno,
            plaintext_line=plaintext_line,
            readname=readname,
            read_seq=read_seq,
            read_phred=read_phred,
            is_mapped=not _check_bit_flag(map_flag, SamFlags.SegmentUnmapped.value),
            is_reverse_complemented=_check_bit_flag(map_flag, SamFlags.SeqReverseComplement.value),
            contig_name=tokens[_SamTags.ContigName.value],
            contig_map_idx=int(tokens[_SamTags.MapPos.value]) - 1,
            cigar=cigar,
            mate_pair=tokens[_SamTags.MatePair.value],
            mate_pos=tokens[_SamTags.MatePos.value],
            template_len=tokens[_SamTags.TemplateLen.value],
            num_aligned_bases=num_aligned_bases,
            num_mismatches=num_mismatches
        )


def num_mismatches_from_xm(match_tag: str):
    head = "XM:i:"
    if not match_tag.startswith(head):
        raise RuntimeError("Expected XM tag to start with `{}`, got `{}`.".format(
            head, match_tag
        ))
    num_mismatches = int(match_tag[len(head):])
    return num_mismatches


class SamFile:
    def __init__(self, file_path: Path, quality_format: str):
        """
        :param file_path:
        :param quality_format: An option (as documented in Bio.SeqIO.QualityIO) for the quality score format.
        """
        self.file_path = file_path
        self.quality_format = quality_format

    @staticmethod
    def _line_is_header(line: str):
        return line[0] == '@'

    def num_mapped_lines(self) -> int:
        with open(self.file_path, 'r') as f:
            return sum(1 for line in f if not self._line_is_header(line))

    def mapped_lines(self) -> Iterator[SamLine]:
        n_lines = 0
        n_mapped_lines = 0
        prev_sam_line: Union[SamLine, None] = None
        with open(self.file_path, 'r') as f:
            for line_idx, line in enumerate(f):
                if self._line_is_header(line):
                    continue

                try:
                    sam_line = SamLine.parse(
                        lineno=line_idx+1,
                        plaintext_line=line,
                        prev_sam_line=prev_sam_line,
                        quality_format=self.quality_format
                    )
                except UnknownNucleotideError as e:
                    raise RuntimeError(
                        f"Encountered unknown nucleotide {e.nucleotide} "
                        f"while reading {self.file_path}, line {line_idx+1}"
                    )
                if sam_line.is_mapped:
                    n_mapped_lines += 1
                    yield sam_line

                prev_sam_line = sam_line
                n_lines += 1
        logger.debug(f"{self.file_path.name} -- Total # SAM lines parsed: {n_lines}; # mapped lines: {n_mapped_lines}")
