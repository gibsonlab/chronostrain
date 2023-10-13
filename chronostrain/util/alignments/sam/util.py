"""
Contains utility functions for all things SAM/BAM related.
"""
from enum import Enum


class SamTags(Enum):
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


class SamFlags(Enum):
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


def has_sam_flag(bit_value: int, flag: SamFlags):
    return bit_value & flag.value == flag.value
