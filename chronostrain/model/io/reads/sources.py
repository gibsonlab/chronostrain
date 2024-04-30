from typing import Iterator, Tuple
from pathlib import Path

import numpy as np
from Bio.SeqIO.QualityIO import phred_quality_from_solexa

from chronostrain.model.reads import SequenceRead, PairedEndRead
from chronostrain.util.io import read_seq_file
from chronostrain.util.sequences import AllocatedSequence, Sequence
from .base import ReadType

from chronostrain.logging import create_logger
logger = create_logger(__name__)


class SampleReadSource(object):
    def __init__(self, name: str):
        self.name = name

    def get_read_depth(self) -> int:
        raise NotImplementedError()

    def reads(self) -> Iterator[SequenceRead]:
        raise NotImplementedError()


class SampleReadSourceSingle(SampleReadSource):
    def __init__(self, read_depth: int, path: Path, quality_format: str, read_type: ReadType, name: str):
        super().__init__(name)
        self.read_depth: int = read_depth
        self.path: Path = path
        self.quality_format: str = quality_format
        self.read_type: ReadType = read_type

    def __str__(self):
        return f"[{self.name} -> {self.path.name}:{self.quality_format}]"

    def __repr__(self):
        return "[{} -> {}:{}:{}]".format(
            self.name,
            self.path, self.quality_format, self.read_type
        )

    def get_read_depth(self) -> int:
        return self.read_depth

    def reads(self) -> Iterator[SequenceRead]:
        for read_id, read_seq, read_qual, read_desc in _parse_fastq(self.path, self.quality_format):
            if self.read_type == ReadType.SINGLE_END:
                yield SequenceRead(read_id, read_seq, read_qual, read_desc)
            elif self.read_type == ReadType.PAIRED_END_1:
                yield PairedEndRead(read_id, read_seq, read_qual, read_desc, is_forward=True)
            elif self.read_type == ReadType.PAIRED_END_2:
                yield PairedEndRead(read_id, read_seq, read_qual, read_desc, is_forward=False)
            else:
                raise NotImplementedError("Unimplemented ReadType instantiation for `{}`".format(self.read_type))


class SampleReadSourcePaired(SampleReadSource):
    def __init__(self, read_depth: int, path_fwd: Path, path_rev: Path, quality_format: str, name: str):
        super().__init__(name)
        self.read_depth: int = read_depth
        self.path_fwd = path_fwd
        self.path_rev = path_rev
        self.quality_format = quality_format

    def __str__(self):
        return f"[{self.name} -> {self.path_fwd.name}+{self.path_rev.name}:{self.quality_format}]"

    def __repr__(self):
        return "[{} -> {}+{}:{}]".format(
            self.name,
            self.path_fwd, self.path_rev, self.quality_format
        )

    def get_read_depth(self) -> int:
        return self.read_depth

    def reads(self) -> Iterator[PairedEndRead]:
        fwd_map = {}
        for prefix_id, read_fwd in self.forward_reads():
            fwd_map[prefix_id] = read_fwd
            yield read_fwd
        for prefix_id, read_rev in self.reverse_reads():
            if prefix_id in fwd_map:
                fwd_mate = fwd_map[prefix_id]
                read_rev.set_mate_pair(fwd_mate)
                fwd_mate.set_mate_pair(read_rev)
            yield read_rev

    def forward_reads(self) -> Iterator[Tuple[str, PairedEndRead]]:
        for read_id, read_seq, read_qual, read_desc in _parse_fastq(self.path_fwd, self.quality_format):
            if not read_id.endswith("/1"):
                raise ValueError("Forward read identifiers are expected to end with `/1`.")
            else:
                mate_prefix = read_id[:-2]
            yield mate_prefix, PairedEndRead(read_id, read_seq, read_qual, read_desc, is_forward=True)

    def reverse_reads(self) -> Iterator[Tuple[str, PairedEndRead]]:
        for read_id, read_seq, read_qual, read_desc in _parse_fastq(self.path_rev, self.quality_format):
            if not read_id.endswith("/2"):
                raise ValueError("Reverse read identifiers are expected to end with `/2`.")
            else:
                mate_prefix = read_id[:-2]
            yield mate_prefix, PairedEndRead(read_id, read_seq, read_qual, read_desc, is_forward=False)


def _parse_fastq(p: Path, quality_format: str) -> Iterator[Tuple[str, Sequence, np.ndarray, str]]:
    for record_idx, record in enumerate(read_seq_file(p, quality_format)):
        # ======= Parse quality scores.
        if (quality_format == "fastq") or (quality_format == "fastq-sanger") or (quality_format == "fastq-illumina"):
            quality = np.array(
                record.letter_annotations["phred_quality"],
                dtype=int
            )
        elif quality_format == "fastq-solexa":
            quality = np.array([
                phred_quality_from_solexa(q) for q in record.letter_annotations["solexa_quality"]
            ], dtype=float)
        else:
            raise ValueError("Unknown quality format `{}`.".format(quality_format))

        yield record.id, AllocatedSequence(str(record.seq)), quality, record.description
