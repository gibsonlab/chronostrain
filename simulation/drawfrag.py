#!/usr/bin/env python
'''Replaces drawfrag C program
From each Fasta sequence in a file, draw random substrings of size k covering it c times, returning a new multi-fasta file with labels
'''

from __future__ import print_function

__version__ = "0.0.1"
import argparse
import os
import sys
import random

from fasta_functions import fasta_reader, check_acgt


def main_not_commandline(args):
    '''All the main code except for the parser'''
    input_file = open(args.input, 'r')
    output_file = open(args.output, 'w')
    taxid_infile = open(args.taxids, 'r')
    gi2taxid_outfile = open(args.gi2taxid, 'w')
    labels_outfile = open(args.labels, 'w')
    k = args.size
    N = args.total

    if args.seed:
        random.seed(args.seed)
    num_seq = len([1 for line in open(args.input) if line.startswith(">")])
    len_seq = [len(line) for _, line in fasta_reader(open(args.input))]
    num_frags = [0 for _ in len_seq]
    s = [l for l in len_seq if l > k]
    ls = sum(s)
    for i in xrange(len(len_seq)):
        if len_seq[i] > k:
            num_frags[i] = round(len_seq[i] * 1.0 * N / ls)
    if sum(num_frags) < N:
        for i in range(len(num_frags)):
            num_frags[i] += (N - sum(num_frags))
            break
    print(num_frags)

    read_num = 0
    i = 0
    for name, seq in fasta_reader(input_file):
        tlabel = taxid_infile.readline().rstrip('\n')
        name_parts = name.split('|')
        firstname = name_parts[0]
        if len(name_parts) > 7:
            firstname = '|'.join([name_parts[3], name_parts[4].strip(':'), name_parts[6], name_parts[7]])
        frag_c = 0
        #	print(len(seq))
        # desired_coverage = c[i] * len(seq)
        if len(seq) < k:
            pass
        else:
            try_num = 0
            while frag_c < num_frags[i]:
                try_num = try_num + 1
                pos = random.randint(0, len(seq) - k)
                sample = seq[pos:pos + k]
                if args.atgc and not check_acgt(sample):
                    pass
                else:
                    frag_c += 1
                    read_num = read_num + 1
                    output_file.write(">{}\n".format(str(read_num) + '|' + firstname))
                    output_file.write("{}\n".format(sample))
                    gi2taxid_outfile.write("{}\t{}\n".format(firstname, tlabel))
                    labels_outfile.write("{}\n".format(tlabel))
                if try_num > 10 * len(seq):
                    break
        i += 1


def main(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('-i', '--input', help='file containing input sequences (fasta or fastq) [required]')
    parser.add_argument('-l', '--size', help='size of drawn items [required]', type=int)
    parser.add_argument('-t', '--taxids',
                        help='one-column file containing the taxid of each input sequence] [required]')
    parser.add_argument('-c', '--coverage', help='mean coverage value for drawing fragments [required]', type=float)
    parser.add_argument('-N', '--total', help='total number of fragments to be drawn [required]', type=int)
    parser.add_argument('-g', '--gi2taxid',
                        help='output gi2taxids file: two-column file containing genome ids and taxids of the drawn fragments [required]')
    parser.add_argument('-s', '--seed',
                        help='value used to initialize the random seed (to use for reproducibility purposes; if not set, will be randomly initialized by Python',
                        type=int)
    parser.add_argument('-o', '--output', help='output sequence file [required]')
    parser.add_argument('--atgc', help='draw fragments made of ATCG only', action='store_true')
    parser.add_argument('-y', '--labels',
                        help='output labels file: one-column file containing taxids of the drawn fragments [required]')

    args = parser.parse_args(argv)
    print(args)
    main_not_commandline(args)


if __name__ == "__main__":
    main(sys.argv[1:])
