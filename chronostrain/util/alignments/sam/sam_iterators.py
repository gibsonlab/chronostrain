from typing import Iterator
from .util import *


def cull_repetitive_templates(sam_lines: Iterator[str]) -> Iterator[str]:
    """
    Only yields alignments so that each (query, database subseq) pair is unique.
    The database subsequence is determined purely based on a combination of the CIGAR string and MD:Z tags.
    """
    current_query = ''
    template_identifiers = set()
    for line in sam_lines:
        line = line.rstrip()
        if sam_line_is_header(line):
            yield line

        tokens = line.rstrip().split('\t')
        read_id = tokens[SamTags.ReadName.value]
        if read_id != current_query:
            # Reset; we are reading the next group of alignments for a new read.
            current_query = read_id
            template_identifiers = set()

        cigar_string = tokens[SamTags.Cigar.value]

        # index 11 and onwards are optional tags.
        prefix = 'MD:Z'
        reference_bases = None
        for optional_tag in tokens[11:]:
            if optional_tag.startswith(prefix):
                reference_bases = optional_tag[len(prefix):]
                break
        if reference_bases is None:
            raise ValueError(f"Expected optional tag `MD:Z`, but not found. (read: {read_id})")

        identifier = (read_id, cigar_string, reference_bases)
        if identifier not in template_identifiers:
            template_identifiers.add(identifier)
            yield line
