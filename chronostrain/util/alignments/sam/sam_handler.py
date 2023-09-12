from pathlib import Path
from typing import List, Iterator, Union, Tuple

import numpy as np

from chronostrain.util.quality import ascii_to_phred
from .cigar import CigarElement, parse_cigar
from chronostrain.util.sequences import Sequence, AllocatedSequence, UnknownNucleotideError

from chronostrain.logging import create_logger
from .util import *
logger = create_logger(__name__)


class SamLine:
    def __init__(self,
                 lineno: int,
                 plaintext_line: str,
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

        :param lineno: The line number in the .sam file corresponding to this instance.
        :param plaintext_line: The raw line read from the .sam file.
        """
        self.lineno = lineno
        self.line = plaintext_line

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

    @property
    def optional_fields(self) -> Iterator[Tuple[str, str, str]]:
        for token in self.line.strip().split('\t')[11:]:
            if token.startswith("SA:"):
                continue  # these report chimeric alignments.
            try:
                tag_name, tag_type, tag_value = token.split(':')
                yield tag_name, tag_type, tag_value
            except ValueError:
                logger.warning("Couldn't parse optional token string {}".format(token))

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

        readname = tokens[SamTags.ReadName.value]
        map_flag = int(tokens[SamTags.MapFlag.value])
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
            read_seq = AllocatedSequence(tokens[SamTags.Read.value])
            read_phred = ascii_to_phred(tokens[SamTags.Quality.value], quality_format)

        cigar = parse_cigar(tokens[SamTags.Cigar.value])

        return SamLine(
            lineno=lineno,
            plaintext_line=plaintext_line,
            readname=readname,
            read_seq=read_seq,
            read_phred=read_phred,
            is_mapped=not has_sam_flag(map_flag, SamFlags.SegmentUnmapped),
            is_reverse_complemented=has_sam_flag(map_flag, SamFlags.SeqReverseComplement),
            contig_name=tokens[SamTags.ContigName.value],
            contig_map_idx=int(tokens[SamTags.MapPos.value]) - 1,
            cigar=cigar,
            mate_pair=tokens[SamTags.MatePair.value],
            mate_pos=tokens[SamTags.MatePos.value],
            template_len=tokens[SamTags.TemplateLen.value]
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

    def num_lines(self) -> int:
        with open(self.file_path, 'r') as f:
            return sum(1 for line in f if not sam_line_is_header(line))

    def mapped_lines(self) -> Iterator[SamLine]:
        prev_sam_line: Union[SamLine, None] = None
        with open(self.file_path, 'r') as f:
            for line_idx, line in enumerate(f):
                if sam_line_is_header(line):
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
                    yield sam_line

                prev_sam_line = sam_line
