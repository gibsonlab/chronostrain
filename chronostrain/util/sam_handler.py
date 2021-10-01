import enum
import re
from typing import List, Iterable, Dict, Union

import numpy as np
from Bio import SeqIO
from chronostrain.util.sequences import complement_seq

from .quality import ascii_to_phred


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
                 reference_sequences: Dict[str, str],
                 samline_prev: Union["SamLine", None],
                 quality_format: str):
        """
        Parse the line using the provided reference.

        :param plaintext_line: A raw SAM file (Tab-separated) entry.
        :param reference_sequences: A dictionary mapping reference (marker) IDS to their sequences.
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
        self.cigar: str = self.line[_SamTags.Cigar.value]
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

        if self.is_mapped:
            self.fragment = self._parse_fragment(reference_sequences)

    def _parse_fragment(self, reference_sequences: Dict[str, str]):
        """
        Finds the fragment in the reference associated with this mapped line

        :param Dict: A mapping from contig names to their respective sequences given by the reference
            multifasta
        :return: The associated fragment as a string
        """

        map_pos_int = int(self.map_pos_str)
        split_cigar = re.findall('\d+|\D+', self.cigar)

        start_clip = 0
        end_clip = 0

        '''
        S and H represent soft and hard clipping respectively. These therefore may occur only at the
        beginning or end of a CIGAR string and represent an offset in starting/ending index into the reference.
        The number of bases clipped precedes the letter demarkation
        '''
        if split_cigar[1] == "S" or split_cigar[1] == "H":
            start_clip = int(split_cigar[0])
        if split_cigar[-1] == 'S' or split_cigar[-1] == "H":
            end_clip = int(split_cigar[-2])

        ref_seq = reference_sequences[self.contig_name]

        start_frame = map_pos_int - start_clip - 1
        end_frame = start_frame + self.read_len

        if start_frame < 0:
            start_frame = 0

        frag = ref_seq[start_frame:end_frame]

        if not self.is_reverse_complemented:
            return frag
        else:
            return complement_seq(frag[::-1])

    def __str__(self):
        return "SamLine(L={lineno}):{tokens}".format(
            lineno=self.lineno,
            tokens=self.line
        )

    def __repr__(self):
        return self.__str__()


class SamHandler:
    def __init__(self, file_path, reference_path, quality_format):
        self.file_path = file_path
        self.reference_path = reference_path
        self.reference_sequences = self.get_multifasta_sequences()

        self.header = []
        self.contents: List[SamLine] = []
        with open(file_path, 'r') as f:
            prev_sam_line = None
            for line_idx, line in enumerate(f):
                if line[0] == '@':
                    self.header.append(line)
                    continue
                sam_line = SamLine(line_idx + 1, line, self.reference_sequences, prev_sam_line, quality_format)
                if sam_line.is_mapped:
                    self.contents.append(sam_line)
                prev_sam_line = sam_line

    def mapped_lines(self) -> Iterable[SamLine]:
        yield from self.contents

    def get_multifasta_sequences(self) -> Dict[str, str]:
        """
        Returns a dictionary mapping Each Record ID (typically a marker identifier) to the sequence.
        """
        reference_sequences = {}
        for record in SeqIO.parse(self.reference_path, format="fasta"):
            reference_sequences[str(record.id)] = str(record.seq)
        return reference_sequences
