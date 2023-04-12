import re
from typing import Tuple, Union
from chronostrain.util.sequences import AllocatedSequence


class PrimerNotFoundError(BaseException):
    def __init__(self, seq_name: str, forward: str, reverse: str):
        super().__init__(
            f"Primer <{forward}>--<{reverse}> not found in {seq_name}."
        )
        self.seq_name = seq_name
        self.forward_primer = forward
        self.reverse_primer = reverse


def regex_match_primers(seq: str, seq_name: str, forward: str, reverse: str, hit_max_len: int) -> Tuple[int, int]:
    forward_primer_regex = _parse_fasta_regex(forward)
    reverse_primer_regex = AllocatedSequence(reverse).revcomp_nucleotides()

    result = _find_primer_match(seq, forward_primer_regex, reverse_primer_regex, hit_max_len)
    if result is None:
        raise PrimerNotFoundError(seq_name, forward, reverse)
    return result


def _parse_fasta_regex(sequence: str):
    fasta_translation = {
        'R': '[AG]', 'Y': '[CT]', 'K': '[GT]',
        'M': '[AC]', 'S': '[CG]', 'W': '[AT]',
        'B': '[CGT]', 'D': '[AGT]', 'H': '[ACT]',
        'V': '[ACG]', 'N': '[ACGT]', 'A': 'A',
        'C': 'C', 'G': 'G', 'T': 'T'
    }
    sequence_regex = ''
    for char in sequence:
        sequence_regex += fasta_translation[char]
    return sequence_regex


def _find_primer_match(seq: str, forward_regex: str, reverse_regex: str, hit_max_len: int) -> Union[None, Tuple[int, int]]:
    # noinspection PyTypeChecker
    best_hit: Tuple[int, int] = None
    best_match_len = hit_max_len

    forward_matches = list(re.finditer(forward_regex, seq))
    reverse_matches = list(re.finditer(reverse_regex, seq))

    for forward_match in forward_matches:
        for reverse_match in reverse_matches:
            match_length = reverse_match.end() - forward_match.start()
            if best_match_len > match_length > 0:
                best_hit = (forward_match.start(), reverse_match.end())
                best_match_len = match_length
    return best_hit