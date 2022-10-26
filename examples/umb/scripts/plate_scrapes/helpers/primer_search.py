import argparse
from dataclasses import dataclass
from typing import Iterator, Tuple, List
from pathlib import Path

import Bio
import Bio.SeqIO
from Bio.Seq import Seq


class NoSequenceMatches(Exception):
    pass


class PairedPrimer(object):
    def __init__(self, gene_name: str, forward: Seq, reverse: Seq):
        self.gene_name = gene_name
        self.forward = forward
        self.reverse = reverse

    def matches(self, query: Seq) -> Iterator[Tuple[int, int]]:
        for start, end in self._match_helper(query):
            assert start < end
            yield start, end

    def _match_helper(self, query: Seq) -> Iterator[Tuple[int, int]]:
        try:
            start_pos = self.single_match_primer(self.forward, query) + len(self.forward)
            end_pos = self.single_match_primer(self.reverse.reverse_complement(), query)
            yield start_pos, end_pos
        except NoSequenceMatches:
            pass

        try:
            start_pos = self.single_match_primer(self.reverse, query) + len(self.reverse)
            end_pos = self.single_match_primer(self.forward.reverse_complement(), query)
            yield start_pos, end_pos
        except NoSequenceMatches:
            pass

    @staticmethod
    def single_match_primer(subseq: Seq, query: Seq) -> int:
        """
        1-indexed position (inclusive) of where the target subsequence starts.
        """
        # Note: this is inefficient; consider using suffix tree instead.
        loc = query.find(subseq)
        if loc < 0:
            raise NoSequenceMatches()
        return loc + 1


def parse_primers(primer_path: Path) -> Iterator[PairedPrimer]:
    records = list(Bio.SeqIO.parse(primer_path, "fasta"))
    for i in range(len(records) // 2):
        fwd = records[2 * i]
        rev = records[1 + 2 * i]
        gene_name_fwd = fwd.id.split('_')[0]
        gene_name_rev = rev.id.split('_')[0]
        if gene_name_fwd != gene_name_rev:
            raise ValueError(f"Got a non-matching primer pair {fwd.id} and {rev.id}.")

        yield PairedPrimer(gene_name_fwd, fwd.seq, rev.seq)


@dataclass
class Hit(object):
    reference_id: str
    reference_len: int
    query_id: str
    start_pos: int
    end_pos: int


def search_scaffolds(scaffold_fasta: Path, primers: List[PairedPrimer]) -> Iterator[Hit]:
    for scaffold in Bio.SeqIO.parse(scaffold_fasta, "fasta"):
        for primer in primers:
            for start_pos, end_pos in primer.matches(scaffold.seq):
                yield Hit(
                    reference_id=scaffold.id,
                    reference_len=len(scaffold.seq),
                    query_id=primer.gene_name,
                    start_pos=start_pos,
                    end_pos=end_pos
                )


def save_hits(hits: List[Hit], out_path: Path):
    with open(out_path, 'w') as f:
        print('\t'.join(['Ref', 'RefLen', 'Query', 'Start', 'End', 'HitLength']), file=f)
        for hit in hits:
            print('\t'.join([
                hit.reference_id, str(hit.reference_len),
                hit.query_id,
                str(hit.start_pos), str(hit.end_pos),
                str(hit.end_pos - hit.start_pos + 1)
            ]), file=f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scaffold-path', dest='scaffold_path', type=str, required=True)
    parser.add_argument('-p', '--primer-path', dest='primer_path', type=str, required=True)
    parser.add_argument('-o', '--out-path', dest='out_path', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    scaffold_path = Path(args.scaffold_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    primers = list(parse_primers(Path(args.primer_path)))
    print("Got primers for: [{}]".format(
        ','.join(p.gene_name for p in primers)
    ))

    hits = list(search_scaffolds(scaffold_path, primers))
    print("Found {} hits.".format(len(hits)))

    save_hits(hits, out_path)


if __name__ == "__main__":
    main()
