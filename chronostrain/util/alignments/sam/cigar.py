import re
from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
from chronostrain.util.sequences import *


class CigarOp(Enum):
    ALIGN = "M"
    INSERTION = "I"
    DELETION = "D"
    SKIPREF = "N"
    CLIPSOFT = "S"
    CLIPHARD = "H"
    PADDING = "P"
    MATCH = "="
    MISMATCH = "X"


cigar_op_dict = {op.value: op for op in CigarOp}
pysam_ordering = [cigar_op_dict[c] for c in 'MIDNSHP=X']


def parse_cigar_op(token: str) -> CigarOp:
    try:
        return cigar_op_dict[token]
    except KeyError:
        raise ValueError("Unknown cigar token `{}`".format(token))


@dataclass
class CigarElement(object):
    op: CigarOp
    num: int


def parse_cigar(cigar: str) -> List[CigarElement]:
    if cigar == "*":
        return []

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


def generate_cigar(ref_align: np.ndarray, query_align: np.ndarray) -> str:
    assert len(ref_align) == len(query_align)
    assert np.sum((ref_align == bytes_GAP) & (query_align == bytes_GAP)) == 0

    cigar_elements: List[CigarElement] = []

    def append_cigar(op: CigarOp):
        if len(cigar_elements) > 0:
            cigar_elements[-1].num += 1
        else:
            cigar_elements.append(CigarElement(op, 1))

    for x, y in zip(ref_align, query_align):
        if x == bytes_GAP:
            append_cigar(CigarOp.INSERTION)
        elif y == bytes_GAP:
            append_cigar(CigarOp.DELETION)
        else:
            append_cigar(CigarOp.ALIGN)

    return "".join(
        f"{cigar_el.num}{cigar_el.op.value}"
        for cigar_el in cigar_elements
    )
