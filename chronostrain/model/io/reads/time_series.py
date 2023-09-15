from pathlib import Path
from typing import List, Iterator, Dict, Tuple, Optional, Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.model.reads import SequenceRead
from chronostrain.util.filesystem import convert_size
from chronostrain.logging import create_logger
from .base import ReadType
from .sources import SampleReadSource, SampleReadSourcePaired, SampleReadSourceSingle
logger = create_logger(__name__)


class TimeSliceReads(object):
    def __init__(self,
                 reads: List[SequenceRead],
                 time_point: float,
                 read_depth: int,
                 sources: Optional[List[SampleReadSource]] = None):
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
    def load(time_point: float, sources: List[SampleReadSource]) -> "TimeSliceReads":
        """
        Creates an instance of TimeSliceReads from the specified file path.

        :param time_point: The timepoint that this source corresponds to.
        :param sources: A List of SampleReadSource instances pointing to the files on disk.
        :return:
        """
        reads = [
            read
            for source in sources
            for read in source.reads()
        ]

        logger.debug(f"(t = {time_point}) Loaded {len(reads)} reads from {len(sources)} fastQ files.")
        total_read_depth = sum(src.get_read_depth() for src in sources)
        return TimeSliceReads(reads, time_point, total_read_depth, sources=sources)

    def get_read(self, read_id: str) -> SequenceRead:
        return self._ids_to_reads[read_id]

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
    def load(all_sources: List[Tuple[float, List[SampleReadSource]]]) -> 'TimeSeriesReads':
        return TimeSeriesReads([
            TimeSliceReads.load(t, sources_t)
            for t, sources_t in all_sources
        ])

    @staticmethod
    def load_from_file(reads_input: Path) -> 'TimeSeriesReads':
        import pandas as pd
        if not reads_input.exists():
            raise FileNotFoundError(f"Missing file `{reads_input}`")

        if reads_input.suffix.lower() == '.csv':
            input_df = pd.read_csv(reads_input, sep=',')
        elif reads_input.suffix.lower() == '.tsv':
            input_df = pd.read_csv(reads_input, sep='\t')
        else:
            raise ValueError(f"Supported file extensions for input files are (.csv, .tsv). Got: {reads_input.suffix}")

        input_df = input_df.rename(
            columns={
                0: 'T',
                1: 'SampleName',
                2: 'ReadDepth',
                3: 'ReadPath',
                4: 'ReadType',
                5: 'QualityFormat'
            }
        ).astype(
            {
                'T': 'float32',
                'SampleName': 'string',
                'ReadDepth': 'int64',
                'ReadPath': 'string',
                'ReadType': 'string',
                'QualityFormat': 'string'
            }
        )

        time_sources: List[Tuple[float, List[SampleReadSource]]] = []
        for t, t_section in input_df.groupby("T"):
            # float(t) results in a warning since groupby()'s key isn't determined by interpreter.
            # noinspection PyTypeChecker
            t = float(t)
            read_sources = []
            for sample_name, sample_section in t_section.groupby("SampleName"):
                if sample_section.shape[0] == 1:
                    _, row = next(iter(sample_section.iterrows()))
                    # ========= Single file; either single-end read or unpaired end of mate pairs.
                    read_sources.append(
                        SampleReadSourceSingle(
                            read_depth=row['ReadDepth'],
                            path=row['ReadPath'],
                            quality_format=row['QualityFormat'],
                            read_type=ReadType.parse_from_str(row['ReadType'])
                        )
                    )
                elif sample_section.shape[0] == 2:
                    # ========= Mate pair files
                    fwd_path: Union[None, Path] = None
                    rev_path: Union[None, Path] = None
                    read_depth = 0
                    quality_fmt = ''
                    for _, row in sample_section.iterrows():
                        read_depth = row['ReadDepth']
                        quality_fmt = row['QualityFormat']
                        read_type = ReadType.parse_from_str(row['ReadType'])
                        if read_type == ReadType.PAIRED_END_1:
                            fwd_path = Path(row['ReadPath'])
                        elif read_type == ReadType.PAIRED_END_2:
                            rev_path = Path(row['ReadPath'])
                        else:
                            raise ValueError(f"Sample {sample_name} contained 2 files; we expected paired_1 and paired_2 mate pairs; but got {row['ReadType']}.")
                    if fwd_path is None:
                        raise ValueError(f"Sample {sample_name} contained 2 files, but no forward mate pair file.")
                    if rev_path is None:
                        raise ValueError(f"Sample {sample_name} contained 2 files, but no reverse mate pair file.")
                    read_sources.append(
                        SampleReadSourcePaired(
                            read_depth=read_depth, path_fwd=fwd_path, path_rev=rev_path, quality_format=quality_fmt
                        )
                    )
                else:
                    raise ValueError("Sample {} contains {} files; is it neither single- nor paired-end?".format(
                        sample_name,
                        sample_section.shape[0]
                    ))
            time_sources.append((t, read_sources))

        return TimeSeriesReads.load(time_sources)

    def __iter__(self) -> Iterator[TimeSliceReads]:
        for time_slice in self.time_slices:
            yield time_slice

    def __len__(self) -> int:
        return len(self.time_slices)

    def __getitem__(self, idx: int) -> TimeSliceReads:
        return self.time_slices[idx]

    def total_number_reads(self) -> int:
        return sum(len(t) for t in self.time_slices)
