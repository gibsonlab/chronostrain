import os
from pathlib import Path
from typing import List
import torch

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.util.logger import logger
from chronostrain.model.reads import SequenceRead
from chronostrain.util.filesystem import convert_size, get_filesize_bytes


def save_timeslice_to_fastq(
        timeslice_reads: List[SequenceRead],
        out_path: str):
    """
    Save reads taken at a particular timepoint to fastq. A helper function for save_to_fastq.
    """
    records = []
    for i, read in enumerate(timeslice_reads):
        # Code from https://biopython.org/docs/1.74/api/Bio.SeqRecord.html
        record = SeqRecord(Seq(read.nucleotide_content()), id="Read#{}".format(i), description=read.metadata)
        record.letter_annotations["phred_quality"] = read.quality.numpy()
        records.append(record)
    SeqIO.write(records, out_path, "fastq")
    logger.debug("Wrote fastQ file {f}. ({sz})".format(
        f=out_path,
        sz=convert_size(get_filesize_bytes(out_path))
    ))


def save_reads_to_fastq(
        sampled_reads: List[List[SequenceRead]],
        time_points: List[float],
        out_dir: str,
        out_prefix: str) -> List[str]:
    """
    Save the sampled reads to a fastq file, one for each timepoint.
    :return: A list of filenames (without parent directory) to which the reads were saved, in order of input timepoints.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if len(sampled_reads) != len(time_points):
        raise ValueError("Number of time indexed lists of reads should equal number of time points to read at")

    prefix_format = '{}_reads_t{}.fastq'
    total_sz = 0

    read_files = []
    for i, t in enumerate(time_points):
        filename = prefix_format.format(out_prefix, str(t).replace('.', '_'))
        read_files.append(filename)
        out_path = os.path.join(out_dir, filename)
        save_timeslice_to_fastq(sampled_reads[i], out_path)
        total_sz += get_filesize_bytes(out_path)
    # return prefix_format.format(out_prefix, '*')

    logger.info("Reads output successfully to {f}. ({sz} Total)".format(
        f=os.path.join(out_dir, prefix_format.format(out_prefix, '*')),
        sz=convert_size(total_sz)
    ))

    return read_files


def load_fastq_reads(file_paths: List[str], format="fastq") -> List[List[SequenceRead]]:
    """
    Load the files from the specified path structure. The files are loaded in order of filenames, assumed to be sorted
    correctly temporally.

    :param file_paths: A list of files inside base_dir to parse, in order of filenames.
    :param format: A valid format, typically one of `fastq`/`fastq-sanger`, `fastq-solexa` or `fastq-illumina`.
    See https://biopython.org/wiki/SeqIO#file-formats.
    :return: A time-indexed list of SequenceRead instances.
    """

    reads = []  # A time-indexed list of read sets. Each item is itself a list of reads for time t.
    for file_path in file_paths:
        reads_t = []  # A list of reads at a particular time (i.e. the reads in 'file')
        for record in SeqIO.parse(file_path, format):
            quality = torch.tensor(record.letter_annotations["phred_quality"], dtype=torch.int)
            read = SequenceRead(
                seq=str(record.seq),
                quality=quality,
                metadata=record.description
            )
            reads_t.append(read)

        logger.debug("Loaded {r} reads from fastQ file {f}. ({sz})".format(
            r=len(reads_t),
            f=file_path,
            sz=convert_size(get_filesize_bytes(file_path))
        ))

        reads.append(reads_t)

    return reads
