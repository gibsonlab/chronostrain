import bz2
import gzip
from collections import Iterator
from pathlib import Path
from contextlib import contextmanager

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


@contextmanager
def open_check_compression(file_path: Path):
    if file_path.suffix == ".gz":
        f = gzip.open(file_path, "rt")
    elif file_path.suffix == ".bz2":
        f = bz2.open(file_path, "rt")
    else:
        f = open(file_path)

    yield f

    f.close()


def read_seq_file(file_path: Path, file_format: str) -> Iterator[SeqRecord]:
    with open_check_compression(file_path) as handle:
        for record in SeqIO.parse(handle, file_format):
            yield record
