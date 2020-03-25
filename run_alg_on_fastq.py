import sys
from Bio import SeqIO
from model import generative
from algs import model_solver
import numpy as np
import re
import os

""" Description
Executable file that takes as input multiple fastq files, each from the same organism but at different time points,
and then returns predicted strain trajectory.

Time points don't have to be consecutive. Input files should be in the same directory and named like:

{ORGANISM ID}_t{TIME POINT}.fastq

So if we had samples at 3 different time points (say at t=0, t=1, and t=4) for the same orgnaism we would have
the following in the data directory:

org1_t0.fastq
org1_t1.fastq
org1_t4.fastq

"""

arguments = sys.argv

data_dir = arguments[0]
file_prefix = arguments[1]
num_timepoints = arguments[2]
algorithm_name = arguments[3]


# debugging
# ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/sequence_read/"
# Downloaded the first file in above URL. Copied it three times, naming them
# "org1_t0.fastq", "org1_t1.fastq", "org1_t2.fastq"
# and placing in "data/" folder.
data_dir = "data/"
file_prefix = "org1"
num_timepoints = 3
algorithm_name = "EM"
###


reads = [] # A time-indexed list of read sets. Each entry is itself a list of reads for time t.

timepoint_to_reads = {}

pattern1 = re.compile("^" + file_prefix)
files = [file for file in os.listdir(data_dir) if pattern1.search(file) ]

pattern2 = re.compile("t[0-9]+\.fastq$")
files_suffix = [pattern2.search(file).group(0) for file in files]

pattern3 = re.compile("[0-9]+")
timepoints = [int(pattern3.search(suffix).group(0)) for suffix in files_suffix]


# Parse the reads (include quality)
for time in timepoints:
    reads_t = []

    for record in SeqIO.parse(data_dir + file_prefix + "_t" + str(time) + ".fastq", "fastq"):

        read = generative.SequenceRead(str(record.seq),
                                       np.asanyarray(record.letter_annotations["phred_quality"]),
                                       "")
        reads_t.append(read)

    timepoint_to_reads[time] = reads_t

# Order the reads by time point
for time in sorted(timepoint_to_reads):
    reads.append(timepoint_to_reads[time])

# TODO: Make set model parameters here
model = generative.GenerativeModel()

# Run algorithm
if algorithm_name == "EM":
    print(model_solver.em_estimate(model, reads, tol=1e-10, iters=10000))
elif algorithm_name == "VI":
    print(model_solver.variational_learn(model, reads, tol=1e-10))
else:
    # Other algs?
    pass
