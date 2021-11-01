import enum
import re
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
                 plaintext_prevline: Union[str, None],
                 quality_format: str):
        """
        Parse the line using the provided reference.

        :param lineno: The line number in the .sam file corresponding to this instance.
        :param plaintext_line: The raw line read from the .sam file.
        :param plaintext_prevline: The previous instance of SamLine corresponding to the previously parsed line.
        :param quality_format: An option (as documented in Bio.SeqIO.QualityIO) for the quality score format.
        """
        self.lineno = lineno
        self.line = plaintext_line.strip().split('\t')

        self.readname: str = self.line[_SamTags.ReadName.value]
        self.map_flag = int(self.line[_SamTags.MapFlag.value])
        self.is_mapped: bool = not _check_bit_flag(self.map_flag, SamFlags.SegmentUnmapped.value)
        self.is_reverse_complemented = _check_bit_flag(self.map_flag, SamFlags.SeqReverseComplement.value)
        self.contig_name: str = self.line[_SamTags.ContigName.value]
        self.map_pos_str: str = self.line[_SamTags.MapPos.value]
        self.map_quality: str = self.line[_SamTags.MapQuality.value]
        self.cigar_str: str = self.line[_SamTags.Cigar.value]
        self.mate_pair: str = self.line[_SamTags.MatePair.value]
        self.mate_pos: str = self.line[_SamTags.MatePos.value]
        self.template_len: str = self.line[_SamTags.TemplateLen.value]

        is_secondary_alignment = _check_bit_flag(self.map_flag, SamFlags.SecondaryAlignment.value)
        if is_secondary_alignment:
            if plaintext_prevline is None:
                raise RuntimeError(
                    "Unexpected SAM file output. Line ({lineno}) was flagged as a secondary alignment, "
                    "but this was the first line to be parsed.".format(
                        lineno=self.lineno
                    )
                )
            prev_line_tokens = plaintext_prevline.strip().split('\t')
            if self.readname != prev_line_tokens[_SamTags.ReadName.value]:
                raise RuntimeError(
                    "Unexpected SAM file output. Line ({lineno}) was flagged as a secondary alignment, "
                    "but the previous line described a different input read ID.".format(
                        lineno=self.lineno
                    )
                )
            self.read: str = prev_line_tokens[_SamTags.Read.value]
            self.read_quality: str = prev_line_tokens[_SamTags.Quality.value]

        else:
            self.read: str = self.line[_SamTags.Read.value]
            self.read_quality: str = self.line[_SamTags.Quality.value]

        self.phred_quality: np.ndarray = ascii_to_phred(self.read_quality, quality_format)
        self.optional_tags = {}
        self.percent_identity: Union[float, None] = None
        for optional_tag in self.line[11:]:
            '''
            The MD tag stores information about which bases match to the reference and is necessary
            for determining percent identity
            '''
            # MD tag: shows percent identity
            if optional_tag[:5] == 'MD:Z:':
                self.optional_tags['MD'] = optional_tag[5:]
                self.percent_identity = percent_identity_from_md_tag(self.optional_tags['MD'])

    @property
    def read_len(self) -> int:
        return len(self.read)

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


def percent_identity_from_md_tag(tag: str):
    """
    Calculate the percent identity from a clipped MD tag. Three types of subsequences are read:
    (1) Numbers represent the corresponding amount of sequential matches
    (2) Letters represent a mismatch and two sequential mismatches are separated by a 0
    (3) A ^ represents a deletion and will be followed by a sequence of consecutive letters
        corresponding to the bases missing
    Dividing (1) by (1)+(2)+(3) will give matches/clipped_length, or percent identity
    """

    '''
    Splits on continuous number sequences.
    '5A0C61^G' -> ['5', 'A', '0', 'C', '61', '^G']
    Which would mean 5 correct bases, two incorrect, 61 correct, then one deleted base.
    Sequential incorrect bases are always split by a 0.
    '''
    split_md = re.findall(r'\d+|\D+', tag)
    total_clipped_length = 0
    total_matches = 0
    for sequence in split_md:
        if sequence.isnumeric():  # (1)
            total_clipped_length += int(sequence)
            total_matches += int(sequence)
        else:
            if sequence[0] == '^':  # (3)
                total_clipped_length += len(sequence) - 1
            elif len(sequence) == 1:  # (2)
                total_clipped_length += 1
            else:
                logger.warning("Unrecognized sequence in MD tag: " + sequence)
    return total_matches / total_clipped_length


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
            prev_line = None
            for line_idx, line in enumerate(f):
                if line[0] == '@':
                    continue

                sam_line = SamLine(line_idx + 1, line, prev_line, self.quality_format)
                if sam_line.is_mapped:
                    n_mapped_lines += 1
                    yield sam_line
                prev_line = line
                n_lines += 1
        logger.debug(f"{self.file_path.name} -- Total # SAM lines parsed: {n_lines}; # mapped lines: {n_mapped_lines}")
