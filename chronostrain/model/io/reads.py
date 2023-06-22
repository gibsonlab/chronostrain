from pathlib import Path
from typing import List, Iterator, Dict, Tuple, Optional
from enum import Enum, auto
import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.QualityIO import phred_quality_from_solexa

from chronostrain.model.reads import SequenceRead, PairedEndRead
from chronostrain.util.filesystem import convert_size
from chronostrain.util.sequences import AllocatedSequence

from chronostrain.logging import create_logger
from chronostrain.util.io import read_seq_file

logger = create_logger(__name__)


class ReadType(Enum):
    PAIRED_END_1 = auto()
    PAIRED_END_2 = auto()
    SINGLE_END = auto()


def parse_read_type(token: str) -> ReadType:
    if token == "paired_1":
        return ReadType.PAIRED_END_1
    elif token == "paired_2":
        return ReadType.PAIRED_END_2
    elif token == "single":
        return ReadType.SINGLE_END
    else:
        raise ValueError(f"Unrecognized read type token `{token}`.")


class TimeSliceReadSource(object):
    def __init__(self, read_depth: int, path: Path, quality_format: str, read_type: ReadType):
        self.read_depth: int = read_depth
        self.path: Path = path
        self.quality_format: str = quality_format
        self.read_type: ReadType = read_type

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return "<TimeSliceReadSource:{}:{}:{}>".format(
            str(self.path), self.quality_format, self.read_type
        )


class TimeSliceReads(object):
    def __init__(self,
                 reads: List[SequenceRead],
                 time_point: float,
                 read_depth: int,
                 sources: Optional[List[TimeSliceReadSource]] = None):
        self.reads: List[SequenceRead] = reads
        self.time_point: float = time_point
        self.read_depth: int = read_depth
        self._ids_to_reads: Dict[str, SequenceRead] = {read.id: read for read in reads}
        self.sources = sources
        if len(reads) == 0:
            self.min_read_length = float('inf')
        else:
            self.min_read_length = min(len(read) for read in reads)

    def save(self, target_path: Path, quality_format: str = "fastq") -> int:
        """
        Save the reads to a fastq file, whose path is specified by attribute `self.file_path`.

        :return: The number of bytes of the output file.
        """
        records = []
        target_path.parent.mkdir(parents=True, exist_ok=True)

        for i, read in enumerate(self.reads):
            # Code from https://biopython.org/docs/1.74/api/Bio.SeqRecord.html
            record = SeqRecord(Seq(read.seq.nucleotides()), id="Read#{}".format(i), description=read.metadata)
            record.letter_annotations["phred_quality"] = read.quality
            records.append(record)
        SeqIO.write(records, target_path, quality_format)

        file_size = target_path.stat().st_size
        logger.info("Wrote fastQ file {f}. ({sz})".format(
            f=target_path,
            sz=convert_size(file_size)
        ))
        return file_size

    @staticmethod
    def load(time_point: float, sources: List[TimeSliceReadSource]) -> "TimeSliceReads":
        """
        Creates an instance of TimeSliceReads() from the specified file path.

        :param time_point: The timepoint that this source corresponds to.
        :param sources: A List of TimeSliceReadSource instances pointing to the files on disk.
        :return:
        """
        reads = []
        offset = 0
        for src in sources:
            n_reads_in_src = 0
            for record_idx, record in enumerate(read_seq_file(src.path, src.quality_format)):
                if (src.quality_format == "fastq") \
                        or (src.quality_format == "fastq-sanger") \
                        or (src.quality_format == "fastq-illumina"):
                    quality = np.array(
                        record.letter_annotations["phred_quality"],
                        dtype=int
                    )
                elif src.quality_format == "fastq-solexa":
                    quality = np.array([
                        phred_quality_from_solexa(q) for q in record.letter_annotations["solexa_quality"]
                    ], dtype=float)
                else:
                    raise ValueError("Unknown quality format `{}`.".format(src.quality_format))

                if src.read_type == ReadType.SINGLE_END:
                    read = SequenceRead(
                        read_id=record.id,
                        read_index=record_idx + offset,
                        seq=AllocatedSequence(str(record.seq)),
                        quality=quality,
                        metadata=record.description
                    )
                elif src.read_type == ReadType.PAIRED_END_1:
                    read = PairedEndRead(record.id, record_idx + offset, AllocatedSequence(str(record.seq)), quality, record.description, forward=True)
                elif src.read_type == ReadType.PAIRED_END_2:
                    read = PairedEndRead(record.id, record_idx + offset, AllocatedSequence(str(record.seq)), quality, record.description, forward=False)
                else:
                    raise NotImplementedError("Unimplemented ReadType instantiation for `{}`".format(src.read_type))
                reads.append(read)
                n_reads_in_src += 1
            offset += n_reads_in_src

            logger.debug(
                "Loaded {r} reads from fastQ file {f}. ({sz})".format(
                    r=n_reads_in_src,
                    f=src.path,
                    sz=convert_size(src.path.stat().st_size)
                )
            )

        logger.debug(f"(t = {time_point}) Loaded {len(reads)} reads from {len(sources)} fastQ files.")
        total_read_depth = sum(src.read_depth for src in sources)
        return TimeSliceReads(reads, time_point, total_read_depth, sources=sources)

    def get_read(self, read_id: str) -> SequenceRead:
        try:
            return self._ids_to_reads[read_id]
        except KeyError as e:
            i = 0
            for k in self._ids_to_reads.keys():
                i += 1
                if i == 10:
                    break
            raise e

    def contains_read(self, read_id: str) -> bool:
        return read_id in self._ids_to_reads

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

    @property
    def min_read_length(self) -> int:
        return min(time_slice.min_read_length for time_slice in self.time_slices)

    def save(self, out_dir: Path):
        """
        Save the sampled reads to a fastq file, one for each timepoint.
        """
        out_dir.mkdir(exist_ok=True, parents=True)
        total_sz = 0
        for time_slice in self.time_slices:
            total_sz += time_slice.save(out_dir / f"reads_{time_slice.time_point}.fastq", "fastq")

        logger.info("Reads output successfully. ({sz} Total)".format(
            sz=convert_size(total_sz)
        ))

    @staticmethod
    def load(time_points: List[float],
             sources: List[List[TimeSliceReadSource]]) -> 'TimeSeriesReads':
        if len(time_points) != len(sources):
            raise ValueError("Number of time points ({}) do not match number of read sources. ({})".format(
                len(time_points), len(sources)
            ))

        return TimeSeriesReads([
            TimeSliceReads.load(t, src_t)
            for t, src_t in zip(time_points, sources)
        ])

    @staticmethod
    def load_from_csv(csv_path: Path) -> 'TimeSeriesReads':
        import csv
        time_points_to_reads: Dict[float, List[Tuple[int, Path, ReadType, str]]] = {}
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required file `{str(csv_path)}`")

        with open(csv_path, "r") as f:
            if csv_path.suffix == '.csv':
                input_specs = csv.reader(f, delimiter=',', quotechar='"')
            elif csv_path.suffix == '.tsv':
                input_specs = csv.reader(f, delimiter='\t', quotechar='"')
            else:
                raise ValueError(f"File extension `{csv_path.suffix}` not recognized from input file `{csv_path}`")
            for row in input_specs:
                time_point = float(row[0])
                read_depth = int(row[1])
                read_path = Path(row[2])
                read_type = parse_read_type(row[3])
                quality_fmt = row[4]

                if not read_path.is_absolute():
                    read_path = csv_path.parent / read_path
                if not read_path.exists():
                    raise FileNotFoundError(
                        "The input specification `{}` pointed to `{}`, which does not exist.".format(
                            str(csv_path),
                            read_path
                        ))

                if time_point not in time_points_to_reads:
                    time_points_to_reads[time_point] = []

                time_points_to_reads[time_point].append((read_depth, read_path, read_type, quality_fmt))

        time_points = sorted(time_points_to_reads.keys(), reverse=False)
        logger.info("Found timepoints: {}".format(time_points))

        time_slice_sources = [
            [
                TimeSliceReadSource(read_depth, read_path, quality_fmt, read_type)
                for read_depth, read_path, read_type, quality_fmt in time_points_to_reads[t]
            ]
            for t in time_points
        ]

        return TimeSeriesReads.load(time_points, time_slice_sources)

    def __iter__(self) -> Iterator[TimeSliceReads]:
        for time_slice in self.time_slices:
            yield time_slice

    def __len__(self) -> int:
        return len(self.time_slices)

    def __getitem__(self, idx: int) -> TimeSliceReads:
        return self.time_slices[idx]

    def total_number_reads(self) -> int:
        return sum(len(t) for t in self.time_slices)
