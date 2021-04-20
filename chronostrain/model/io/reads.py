from pathlib import Path
from typing import List, Optional
import torch

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from . import logger
from chronostrain.model.reads import SequenceRead
from chronostrain.util.filesystem import convert_size


class TimeSliceReads(object):
    def __init__(self, reads: List[SequenceRead], time_point: float, src: Optional[Path] = None):
        self.reads = reads
        self.time_point = time_point
        self.src = src

    def save(self) -> int:
        """
        Save the reads to a fastq file, whose path is specified by attribute `self.file_path`.

        :return: The number of bytes of the output file.
        """
        if self.src is None:
            raise ValueError("Specify the `src` parameter if invoking save() of TimeSliceReads object.")

        records = []
        Path(self.src).parent.mkdir(parents=True, exist_ok=True)

        for i, read in enumerate(self.reads):
            # Code from https://biopython.org/docs/1.74/api/Bio.SeqRecord.html
            record = SeqRecord(Seq(read.nucleotide_content()), id="Read#{}".format(i), description=read.metadata)
            record.letter_annotations["phred_quality"] = read.quality
            records.append(record)
        SeqIO.write(records, self.src, "fastq")

        file_size = self.src.stat().st_size
        logger.info("Wrote fastQ file {f}. ({sz})".format(
            f=self.src,
            sz=convert_size(file_size)
        ))
        return file_size

    @staticmethod
    def load(file_path: Path, time_point: float):
        reads = []
        for record in SeqIO.parse(file_path, "fastq"):
            quality = torch.tensor(record.letter_annotations["phred_quality"], dtype=torch.int)
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
        return TimeSliceReads(reads, time_point, file_path)

    def __iter__(self) -> SequenceRead:
        for read in self.reads:
            yield read

    def __len__(self) -> int:
        return len(self.reads)

    def __getitem__(self, idx: int) -> SequenceRead:
        return self.reads[idx]


class TimeSeriesReads(object):
    def __init__(self, time_slices: List[TimeSliceReads]):
        self.time_slices = time_slices

    def save(self):
        """
        Save the sampled reads to a fastq file, one for each timepoint.
        """
        total_sz = 0
        for time_slice in self.time_slices:
            total_sz += time_slice.save()

        logger.info("Reads output successfully. ({sz} Total)".format(
            sz=convert_size(total_sz)
        ))

    @staticmethod
    def load(time_points: List[float], file_paths: List[Path]):
        if len(time_points) != len(file_paths):
            raise ValueError("Number of time points ({}) do not match number of file paths. ({})".format(
                len(time_points), len(file_paths)
            ))
        time_slices = [
            TimeSliceReads.load(file_path, t)
            for file_path, t in zip(file_paths, time_points)
        ]
        return TimeSeriesReads(time_slices)

    def __iter__(self) -> TimeSliceReads:
        for time_slice in self.time_slices:
            yield time_slice

    def __len__(self) -> int:
        return len(self.time_slices)

    def __getitem__(self, idx: int) -> TimeSliceReads:
        return self.time_slices[idx]
