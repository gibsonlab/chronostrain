import math
from typing import List

import numpy as np
from Bio.SeqIO.QualityIO import phred_quality_from_solexa


def _parse_ascii_using_offset(qstr: str, offset: int) -> List[int]:
    """
    Parse ASCII code into ints using `ord`, minus the specified offset.
    Mirrors the implementation found in Bio.SeqIO.QualityIO.
    """
    return [ord(letter) - offset for letter in qstr]


# === Offsets: ord('@') = 64, ord('!') = 33.
def _str_to_phred_fastq_solexa(qstr: str) -> List[float]:
    return [phred_quality_from_solexa(q) for q in _parse_ascii_using_offset(qstr, ord("@"))]


def _str_to_phred_fastq_illumina(qstr) -> List[int]:
    """
    Corresponds to the 'fastq-illumina' option in Bio.SeqIO.QualityIO.
    As stated in the Bio.SeqIO.QualityIO documentation, this is for newer Illumina 1.3-1.7 FASTQ files.
    """
    return _parse_ascii_using_offset(qstr, ord("@"))


def _str_to_phred_fastq_sanger(qstr) -> List[int]:
    """
    Sanger-style phred scores, also used by Illumina 1.8+ (according to Bio.SeqIO.QualityIO docs).
    """
    return _parse_ascii_using_offset(qstr, ord('!'))


def ascii_to_phred(qstr: str, quality_format: str) -> np.ndarray:
    """
    :param qstr: The string of quality scores for a particular read.
    :param quality_format: An option (as documented in Bio.SeqIO.QualityIO) for the quality score format.
    """
    if quality_format == 'fastq':
        return np.array(_str_to_phred_fastq_sanger(qstr), dtype=float)
    elif quality_format == 'fastq-sanger':
        return np.array(_str_to_phred_fastq_sanger(qstr), dtype=float)
    elif quality_format == 'fastq-solexa':
        return np.array(_str_to_phred_fastq_solexa(qstr), dtype=float)
    elif quality_format == 'fastq-illumina':
        return np.array(_str_to_phred_fastq_illumina(qstr), dtype=float)
    else:
        raise NotImplementedError("Unknown quality_format input `{}`.".format(quality_format))


def ascii_pysam_to_phred(qlist: List[int]) -> np.ndarray:
    return np.array(qlist, dtype=float)


def phred_to_ascii(phred_arr: np.ndarray, quality_format: str) -> str:
    if quality_format == 'fastq':
        return "".join(chr(math.floor(i)) for i in ord('!') + phred_arr)
    else:
        raise NotImplementedError("Unknown quality_format input `{}`.".format(quality_format))

