from pathlib import Path
from typing import List, Optional, Union, Iterable, Iterator, Dict
import numpy as np

import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.QualityIO import phred_quality_from_solexa

from chronostrain.model.reads import SequenceRead
from chronostrain.util.filesystem import convert_size

from chronostrain.config.logging import create_logger
logger = create_logger(__name__)


class TimeSliceReadSource(object):
    def __init__(self, paths: List[Path], quality_format: str):
        self.paths: List[Path] = paths
        self.quality_format: str = quality_format

    def get_canonical_path(self) -> Path:
        if len(self.paths) != 1:
            raise ValueError("Found {} paths, cannot decide canonical one.".format(len(self.paths)))
        return self.paths[0]

    def __str__(self):
        return self.paths.__str__()

    def __repr__(self):
        return "<TimeSliceReadSource:{}>".format(self.paths.__repr__())


class TimeSliceReads(object):
    def __init__(self,
                 reads: List[SequenceRead],
                 time_point: float,
                 read_depth: int,
                 src: Optional[TimeSliceReadSource] = None):
        self.reads: List[SequenceRead] = reads
        self.time_point: float = time_point
        self.src: Union[TimeSliceReadSource, None] = src
        self.read_depth: int = read_depth
        self._ids_to_reads: Dict[str, SequenceRead] = {read.id: read for read in reads}

    def save(self, quality_format: str) -> int:
        """
        Save the reads to a fastq file, whose path is specified by attribute `self.file_path`.

        :return: The number of bytes of the output file.
        """
        if self.src is None:
            raise ValueError("Specify the `src` parameter if invoking save() of TimeSliceReads object.")

        canonical_path = self.src.get_canonical_path()
        records = []
        canonical_path.parent.mkdir(parents=True, exist_ok=True)

        for i, read in enumerate(self.reads):
            # Code from https://biopython.org/docs/1.74/api/Bio.SeqRecord.html
            record = SeqRecord(Seq(read.nucleotide_content()), id="Read#{}".format(i), description=read.metadata)
            record.letter_annotations["phred_quality"] = read.quality
            records.append(record)
        SeqIO.write(records, canonical_path, quality_format)

        file_size = canonical_path.stat().st_size
        logger.info("Wrote fastQ file {f}. ({sz})".format(
            f=canonical_path,
            sz=convert_size(file_size)
        ))
        return file_size

    @staticmethod
    def read_fastq(fastq_path: Path, quality_format: str) -> Iterator[SeqRecord]:
        for record in SeqIO.parse(fastq_path, quality_format):
            yield record

    @staticmethod
    def read_gzipped_fastq(gz_path: Path, quality_format: str) -> Iterator[SeqRecord]:
        with gzip.open(str(gz_path), "rt") as handle:
            for record in SeqIO.parse(handle, quality_format):
                yield record

    @staticmethod
    def load(src: TimeSliceReadSource, read_depth: int, time_point: float) -> "TimeSliceReads":
        """
        Creates an instance of TimeSliceReads() from the specified file path.

        :param src: A TimeSliceReadSource instance pointing to the files on disk.
        :param read_depth: The read depth (total number of reads in the experiment) for this particular timepoint.
        :param time_point: The timepoint that this source corresponds to.
        :return:
        """
        reads = []
        quality_format = src.quality_format
        for file_path in src.paths:
            if file_path.suffix == '.gz':
                read_fn = TimeSliceReads.read_gzipped_fastq
            else:
                read_fn = TimeSliceReads.read_fastq

            for record in read_fn(file_path, quality_format):
                if (quality_format == "fastq") \
                        or (quality_format == "fastq-sanger") \
                        or (quality_format == "fastq-illumina"):
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
                read = SequenceRead(
                    read_id=record.id,
                    seq=str(record.seq),
                    quality=quality,
                    metadata=record.description
                )
                reads.append(read)

            logger.debug("Loaded {r} reads from fastQ file {f}. ({sz})".format(
                r=len(reads),
                f=file_path,
                sz=convert_size(file_path.stat().st_size)
            ))
        return TimeSliceReads(reads, time_point, read_depth, src)

    def get_read(self, read_id: str) -> SequenceRead:
        return self._ids_to_reads[read_id]

    def __iter__(self) -> Iterator[SequenceRead]:
        for read in self.reads:
            yield read

    def __len__(self) -> int:
        return len(self.reads)

    def __getitem__(self, idx: int) -> SequenceRead:
        return self.reads[idx]


class TimeSeriesReads(object):
    def __init__(self, time_slices: List[TimeSliceReads]):
        self.time_slices = time_slices

    def save(self, quality_format: str):
        """
        Save the sampled reads to a fastq file, one for each timepoint.
        """
        total_sz = 0
        for time_slice in self.time_slices:
            total_sz += time_slice.save(quality_format)

        logger.info("Reads output successfully. ({sz} Total)".format(
            sz=convert_size(total_sz)
        ))

    @staticmethod
    def load(time_points: List[float],
             read_depths: List[int],
             source_entries: List[Iterable[Path]],
             quality_format: str):
        if len(time_points) != len(source_entries):
            raise ValueError("Number of time points ({}) do not match number of read sources. ({})".format(
                len(time_points), len(source_entries)
            ))

        time_slice_sources = [
            TimeSliceReadSource(list(file_paths), quality_format)
            for file_paths in source_entries
        ]

        return TimeSeriesReads([
            TimeSliceReads.load(src, read_depth_t, t)
            for src, read_depth_t, t in zip(time_slice_sources, read_depths, time_points)
        ])

    def __iter__(self) -> Iterator[TimeSliceReads]:
        for time_slice in self.time_slices:
            yield time_slice

    def __len__(self) -> int:
        return len(self.time_slices)

    def __getitem__(self, idx: int) -> TimeSliceReads:
        return self.time_slices[idx]
