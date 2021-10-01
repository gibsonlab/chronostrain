import numpy as np
from Bio.SeqIO.QualityIO import phred_quality_from_solexa


def _parse_ascii_using_offset(qstr: str, offset: int):
    """
    Parse ASCII code into ints using `ord`, minus the specified offset.
    Mirrors the implementation found in Bio.SeqIO.QualityIO.
    """
    return np.array([ord(letter) - offset for letter in qstr], dtype=float)


# === Offsets: ord('@') = 64, ord('!') = 33.
def _phred_fastq_solexa(qstr):
    return [phred_quality_from_solexa(q) for q in _parse_ascii_using_offset(qstr, ord("@"))]


def _phred_fastq_illumina(qstr):
    """
    Corresponds to the 'fastq-illumina' option in Bio.SeqIO.QualityIO.
    As stated in the Bio.SeqIO.QualityIO documentation, this is for newer Illumina 1.3-1.7 FASTQ files.
    """
    return _parse_ascii_using_offset(qstr, ord("@"))


def _phred_fastq_sanger(qstr):
    """
    Sanger-style phred scores, also used by Illumina 1.8+ (according to Bio.SeqIO.QualityIO docs).
    """
    return _parse_ascii_using_offset(qstr, ord('!'))


def ascii_to_phred(qstr: str, quality_format: str):
    if quality_format == 'fastq':
        return _phred_fastq_sanger(qstr)
    elif quality_format == 'fastq-sanger':
        return _phred_fastq_sanger(qstr)
    elif quality_format == 'fastq-solexa':
        return _phred_fastq_solexa(qstr)
    elif quality_format == 'fastq-illumina':
        return _phred_fastq_illumina(qstr)
    else:
        raise ValueError("Unknown quality_format input `{}`.".format(quality_format))
