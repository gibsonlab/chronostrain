"""
  model_io.py

  All model-related IO functions are placed here. (e.g. saving/loading models, saving abundances or reads to file.)
"""

import os
import csv
import torch
from pathlib import Path
from typing import List, Tuple

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from chronostrain.config import cfg
from chronostrain.model.bacteria import Population
from chronostrain.model.reads import SequenceRead
from chronostrain.util.io.filesystem import convert_size, get_filesize_bytes
from chronostrain.util.io.logger import logger


def save_abundances_by_path(
        population: Population,
        time_points: List[float],
        abundances: torch.Tensor,
        out_path: str):
    path = Path(out_path)
    parent = Path(path.parent)
    parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        writer.writerow(["T"] + [strain.name for strain in population.strains])
        # for k, abundance in enumerate(abundances):
        for t in range(len(time_points)):
            writer.writerow([time_points[t]] + [x.item() for x in abundances[t]])
    logger.info("Abundances output successfully to {}. ({})".format(
        out_path, convert_size(get_filesize_bytes(out_path))
    ))
    return out_path


def save_abundances(
        population: Population,
        time_points: List[float],
        abundances: torch.Tensor,
        out_dir: str,
        out_filename: str,
):
    """
    Save the time-indexed abundance profile to disk. Output format is CSV.

    :param population: The Population instance containing the strain information.
    :param time_points: The list of time points in the data.
    :param abundances: A T x S tensor containing time-indexed relative abundances profiles.
    :param out_filename: The filename to write to.
    :param out_dir: The directory to specify the path.
    :return: The path/filename for the abundance CSV file.
    """
    if len(population.strains) != len(abundances[0]):
        raise Exception("Length of strains doesn't match length of abundance profile.")
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, out_filename)
    else:
        out_path = out_filename
    return save_abundances_by_path(population, time_points, abundances, out_path)


def load_abundances(file_path: str) -> Tuple[List[int], torch.Tensor, List[str]]:
    """
    Read time-indexed abundances from file.

    :return: (1) A list of time points,
    (2) a time indexed list of abundance profiles,
    (3) the list of relevant accessions.
    """

    time_points = []
    strain_abundances = []
    accessions = []

    with open(file_path, newline='') as f:
        reader = csv.reader(f, quotechar='"')
        for i, row in enumerate(reader):
            if i == 0:
                accessions = [x.replace('"', '').strip() for x in row[1:]]
                continue
            if not row:
                continue
            time_point = row[0]
            abundances = torch.tensor(
                [float(val) for val in row[1:]],
                dtype=torch.double,
                device=cfg.torch_cfg.device
            )
            time_points.append(time_point)
            strain_abundances.append(abundances)
    return time_points, torch.stack(strain_abundances, dim=0), accessions


def save_reads_to_fastq(
        sampled_reads: List[List[SequenceRead]],
        time_points: List[float],
        out_dir: str,
        out_prefix: str):
    """
    Save the sampled reads to a fastq file, one for each timepoint.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if len(sampled_reads) != len(time_points):
        raise ValueError("Number of time indexed lists of reads should equal number of time points to read at")

    prefix_format = '{}_reads_t{}.fastq'
    total_sz = 0
    for i, t in enumerate(time_points):
        filename = prefix_format.format(out_prefix, str(t).replace('.', '_'))
        out_path = os.path.join(out_dir, filename)
        save_timeslice_to_fastq(sampled_reads[i], out_path)
        total_sz += get_filesize_bytes(out_path)
    # return prefix_format.format(out_prefix, '*')

    logger.info("Reads output successfully to {f}. ({sz} Total)".format(
        f=os.path.join(out_dir, prefix_format.format(out_prefix, '*')),
        sz=convert_size(total_sz)
    ))


def save_timeslice_to_fastq(
        timeslice_reads: List[SequenceRead],
        out_path: str):
    """
    Save reads taken at a particular timepoint to fastq. A helper function for save_to_fastq.
    """
    records = []
    for i, read in enumerate(timeslice_reads):
        # Code from https://biopython.org/docs/1.74/api/Bio.SeqRecord.html
        record = SeqRecord(Seq(read.seq), id="Read#{}".format(i), description=read.metadata)
        record.letter_annotations["phred_quality"] = read.quality.numpy()
        records.append(record)
    SeqIO.write(records, out_path, "fastq")
    logger.debug("Wrote fastQ file {f}. ({sz})".format(
        f=out_path,
        sz=convert_size(get_filesize_bytes(out_path))
    ))


def load_fastq_reads(file_paths: List[str]) -> List[List[SequenceRead]]:
    """
    Load the files from the specified path structure. The files are loaded in order of filenames, assumed to be sorted
    correctly temporally.

    :param file_paths: A list of files inside base_dir to parse, in order of filenames.
    :return: A time-indexed list of SequenceRead instances.
    """

    reads = []  # A time-indexed list of read sets. Each item is itself a list of reads for time t.
    for file_path in file_paths:
        reads_t = []  # A list of reads at a particular time (i.e. the reads in 'file')
        for record in SeqIO.parse(file_path, "fastq"):
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
