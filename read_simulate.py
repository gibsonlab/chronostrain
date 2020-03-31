"""
  read_simulate.py
  Run to simulate reads from genomes specified by accession numbers.
"""
import os
import argparse
from scripts.fetch_genomes import fetch_sequences
from util.logger import logger

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

import random
import re
from model import generative, reads

from Bio import SeqIO


_data_dir = "data"

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate reads from genomes.")
    parser.add_argument('-s', '--seed', required=False, type=int, default=31415,
                        help='<Optional> Seed for randomness (for reproducibility).')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> Directory to save the reads.')

    parser.add_argument('-a', '--accession_num', required=True, type = str,
                        help='<Required> Accession number of the species to sample from.')

    parser.add_argument('-t', '--time_points', required=True, type=int, nargs="+",
                        help='<Required> A list of intergers, where each integer represents a time point'
                             'for which to take a sample from. Time points are saved as part of file name.')

    parser.add_argument('-n', '--num_reads', required=True, type = int, nargs='+',
                        help='<Required> Numbers of the reads to sample at each time point. '
                             'Must either be a single integer or a list of integers with length equal to the number of time points.')

    parser.add_argument('-l', '--read_length', required=True, type = int,
                        help='<Required> Length of the reads to sample.')


    parser.add_argument('-p', '--out_prefix', required=False, default='sampled_read',
                        help='<Optional> File prefix for the read files.')
    parser.add_argument('-e', '--extension', required=False, default='txt',
                        help='<Optional> File extension.')

    return parser.parse_args()


def save_to_fastq(sampled_reads, time_points, out_dir, out_prefix):

    if len(sampled_reads) != len(time_points):
        raise ValueError("Number of time indexed lists of reads should equal number of time points to read at")

    for i, t in enumerate(time_points):
        filename = '{}_{}.fastq'.format(out_prefix, t)
        out_path = os.path.join(out_dir, filename)
        save_timeslice_to_fastq(sampled_reads[i], out_path)

def save_timeslice_to_fastq(reads, out_path):
    # Save reads taken at a particular timepoint to fastq.

    records = []
    for i, read in enumerate(reads):

        # https://biopython.org/docs/1.74/api/Bio.SeqRecord.html
        record = SeqRecord(Seq(read.seq), id="Read number " + str(i), description=", a simulated read")
        record.letter_annotations["phred_quality"] = list(read.quality)

        records.append(record)

    SeqIO.write(records, out_path, "fastq")


def sample_reads(accession_number, number_reads, read_length, time_points, seed):

    random.seed(seed)

    # TODO: Use SeqIO to open fasta file? Wasn't working correctly so had to do manually here.
    genome = ""
    filename = _data_dir + "/" + accession_number + ".fasta"
    with open(filename) as file:
        for i, line in enumerate(file):
            genome = re.sub('[^AGCT]+', '', line.split(sep=" ")[-1])

    reads_list = []

    for t in range(len(time_points)):
        reads_at_t = []
        for n in range(len(number_reads)):

            start_index = random.randint(0, len(genome) - read_length)
            sequence = genome[start_index:start_index + read_length]

            phred_score_dist = reads.PhredScoreDistribution(read_length)
            quality = phred_score_dist.create_qvec(phred_score_dist.distribution)

            read = generative.SequenceRead(sequence, quality, metadata="")
            reads_at_t.append(read)

        reads_list.append(reads_at_t)

    return reads_list


def main():
    logger.info("Pipeline for read simulation started.")
    args = parse_args()
    logger.debug("Downloading genomes from NCBI...")
    fetch_sequences()
    logger.debug("Sampling reads...")

    sampled_reads = sample_reads(
        accession_number=args.accession_num,
        number_reads=args.num_reads,
        read_length=args.read_length,
        time_points=args.time_points,
        seed=args.seed
    )
    logger.debug("Saving samples to FastQ file {}.".format(args.out_dir + args.out_prefix))
    save_to_fastq(sampled_reads, args.time_points, args.out_dir, args.out_prefix)
    logger.info("Reads finished sampling.")

if __name__ == "__main__":
    main()

