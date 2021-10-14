import enum
from pathlib import Path
from typing import List, Iterator, Union

import numpy as np

from chronostrain.util.quality import ascii_to_phred
from .cigar import CigarElement, parse_cigar

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


class _MapFlags(enum.Enum):
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
                 samline_prev: Union["SamLine", None],
                 quality_format: str):
        """
        Parse the line using the provided reference.

        :param lineno: The line number in the .sam file corresponding to this instance.
        :param plaintext_line: The raw line read from the .sam file.
        :param samline_prev: The previous instance of SamLine corresponding to the previously parsed line.
        :param quality_format: An option (as documented in Bio.SeqIO.QualityIO) for the quality score format.
        """
        self.lineno = lineno
        self.line = plaintext_line.strip().split('\t')
        self.prev_line = samline_prev

        self.readname: str = self.line[_SamTags.ReadName.value]
        self.map_flag = int(self.line[_SamTags.MapFlag.value])
        self.is_mapped: bool = not _check_bit_flag(self.map_flag, _MapFlags.SegmentUnmapped.value)
        self.is_reverse_complemented = _check_bit_flag(self.map_flag, _MapFlags.SeqReverseComplement.value)
        self.contig_name: str = self.line[_SamTags.ContigName.value]
        self.map_pos_str: str = self.line[_SamTags.MapPos.value]
        self.map_quality: str = self.line[_SamTags.MapQuality.value]
        self.cigar_str: str = self.line[_SamTags.Cigar.value]
        self.mate_pair: str = self.line[_SamTags.MatePair.value]
        self.mate_pos: str = self.line[_SamTags.MatePos.value]
        self.template_len: str = self.line[_SamTags.TemplateLen.value]

        is_secondary_alignment = _check_bit_flag(self.map_flag, _MapFlags.SecondaryAlignment.value)
        if is_secondary_alignment:
            if self.prev_line is None:
                raise RuntimeError(
                    "Unexpected SAM file output. Line ({lineno}) was flagged as a secondary alignment, "
                    "but this was the first line to be parsed.".format(
                        lineno=self.lineno
                    )
                )
            if self.readname != self.prev_line.readname:
                raise RuntimeError(
                    "Unexpected SAM file output. Line ({lineno}) was flagged as a secondary alignment, "
                    "but the previous line described a different input read ID.".format(
                        lineno=self.lineno
                    )
                )
            self.read: str = self.prev_line.read
            self.read_len: int = self.prev_line.read_len
            self.read_quality: str = self.prev_line.read_quality
            self.phred_quality: np.ndarray = self.prev_line.phred_quality
        else:
            self.read: str = self.line[_SamTags.Read.value]
            self.read_len: int = len(self.read)
            self.read_quality: str = self.line[_SamTags.Quality.value]
            self.phred_quality: np.ndarray = ascii_to_phred(self.read_quality, quality_format)

        self.optional_tags = {}
        for optional_tag in self.line[11:]:
            '''
            The MD tag stores information about which bases match to the reference and is necessary
            for determining percent identity
            '''
            if optional_tag[:5] == 'MD:Z:':
                self.optional_tags['MD'] = optional_tag[5:]

    def __str__(self):
        return "SamLine(L={lineno}):{tokens}".format(
            lineno=self.lineno,
            tokens=self.line
        )

    def __repr__(self):
        return self.__str__()

    @property
    def cigar(self) -> List[CigarElement]:
        return parse_cigar(self.cigar_str)


class SamFile:
    def __init__(self, file_path: Path, quality_format: str):
        """
        :param file_path:
        :param quality_format: An option (as documented in Bio.SeqIO.QualityIO) for the quality score format.
        """
        self.file_path = file_path
        self.quality_format = quality_format

    def mapped_lines(self) -> Iterator[SamLine]:
        n_lines = 0
        n_mapped_lines = 0
        with open(self.file_path, 'r') as f:
            prev_sam_line = None
            for line_idx, line in enumerate(f):
                if line[0] == '@':
                    continue

                sam_line = SamLine(line_idx + 1, line, prev_sam_line, self.quality_format)
                if sam_line.is_mapped:
                    n_mapped_lines += 1
                    yield sam_line
                prev_sam_line = sam_line
                n_lines += 1
        logger.debug(f"Total # SAM lines parsed: {n_lines}; # mapped lines: {n_mapped_lines}")
