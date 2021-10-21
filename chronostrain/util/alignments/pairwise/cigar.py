import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class CigarOp(Enum):
    ALIGN = auto()
    INSERTION = auto()
    DELETION = auto()
    SKIPREF = auto()
    CLIPSOFT = auto()
    CLIPHARD = auto()
    PADDING = auto()
    MATCH = auto()
    MISMATCH = auto()


def parse_cigar_op(token: str) -> CigarOp:
    if token == 'M':
        return CigarOp.ALIGN
    elif token == 'I':
        return CigarOp.INSERTION
    elif token == 'D':
        return CigarOp.DELETION
    elif token == 'N':
        return CigarOp.SKIPREF
    elif token == 'S':
        return CigarOp.CLIPSOFT
    elif token == 'H':
        return CigarOp.CLIPHARD
    elif token == 'P':
        return CigarOp.PADDING
    elif token == '=':
        return CigarOp.MATCH
    elif token == 'X':
        return CigarOp.MISMATCH
    else:
        raise ValueError("Unknown cigar token `{}`".format(token))


@dataclass
class CigarElement(object):
    op: CigarOp
    num: int


def parse_cigar(cigar: str) -> List[CigarElement]:
    if cigar == "*":
        raise ValueError('No cigar string to parse. (Cigar string "*" hints at an unmapped read.)')

    tokens = re.findall(r'\d+|\D+', cigar)
    if len(tokens) % 2 != 0:
        raise ValueError("Expected an even number of tokens in cigar string ({}). Got: {}".format(
            cigar, len(tokens)
        ))

    elements = []
    for i in range(0, len(tokens), 2):
        elements.append(CigarElement(
            parse_cigar_op(tokens[i+1]),
            int(tokens[i])
        ))
    return elements
