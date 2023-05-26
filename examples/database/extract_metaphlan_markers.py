from typing import Iterator, Tuple, Set, Dict
from pathlib import Path

import pickle
import bz2
import argparse

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--taxon_label', type=str)
    parser.add_argument('-i', '--input_metaphlan_pkl', type=str)
    parser.add_argument('-o', '--output_index', type=str)
    return parser.parse_args()


class MetaphlanParser(object):
    def __init__(self, metaphlan_pkl_path: Path):
        self.pkl_path = metaphlan_pkl_path
        self.marker_fasta = metaphlan_pkl_path.with_suffix('.fna')
        if not self.marker_fasta.exists():
            raise FileNotFoundError(f"Expected {self.marker_fasta} to exist, but not found.")

    def retrieve_marker_seeds(self, taxon_key: str) -> Iterator[Tuple[str, SeqRecord]]:
        print(f"Searching for marker seeds from MetaPhlAn database: {self.pkl_path.stem}.")
        with bz2.open(self.pkl_path, "r") as f:
            db = pickle.load(f)

        markers = db['markers']
        target_keys = set()
        for marker_key, marker_dict in markers.items():
            if taxon_key in marker_dict['taxon']:
                target_keys.add(marker_key)

        print(f"Target # of markers: {len(target_keys)}")
        for marker_key, seq in self._retrieve_from_fasta(target_keys):
            print(f"Found metaphlan marker ID {marker_key}.")
            record = SeqRecord(seq, id=marker_key, description=self.pkl_path.stem)
            yield marker_key, record

    def _retrieve_from_fasta(self, marker_keys: Set[str]) -> Tuple[str, Seq]:
        remaining = set(marker_keys)
        with open(self.marker_fasta, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                if len(remaining) == 0:
                    break
                if record.id not in remaining:
                    continue

                remaining.remove(record.id)
                yield record.id, record.seq
        if len(remaining) > 0:
            print(f"For some reason, couldn't locate {len(remaining)} markers from Fasta: {remaining}")


def main():
    args = parse_args()
    output_index_path = Path(args.output_index)
    output_index_path.parent.mkdir(exist_ok=True, parents=True)

    parser = MetaphlanParser(Path(args.input_metaphlan_pkl))

    # Extract reference seqs
    target_markers: Dict[str, Path] = {}
    for marker_name, record in parser.retrieve_marker_seeds(args.taxon_label):
        marker_len = len(record.seq)
        print(f"Found marker `{marker_name}` (length {marker_len})")

        fasta_path = output_index_path.parent / f"{marker_name}.fasta"
        SeqIO.write(record, fasta_path, "fasta")

        target_markers[marker_name] = fasta_path

    # Create TSV index.
    with open(output_index_path, "w") as f:
        for marker_name, marker_path in target_markers.items():
            print(
                f"{marker_name}\t{marker_path}\tMetaPhlAn:{parser.pkl_path.stem}",
                file=f
            )
    print(f"Wrote marker seed index {output_index_path}")


if __name__ == "__main__":
    main()
