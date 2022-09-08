# NOTE: python 2.7

import os
import argparse
import csv
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index_refseq', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    return parser.parse_args()


def handle_entry(fp, accession, species, gcf, seq_path):
    if species != 'coli':
        return

    candidates = glob.glob(
        os.path.join(os.path.dirname(seq_path), '{}_*.chrom.gff'.format(gcf))
    )
    if len(candidates) == 0:
        raise RuntimeError("Could not find .gff entry for {}".format(accession))
    elif len(candidates) > 1:
        raise RuntimeError("Multiple .gff entries found for {}".format(accession))

    gff_path = candidates[0]

    fp.write('Genome\t{}\n'.format(accession))
    fp.write('Sequence\t{}\n'.format(seq_path))
    fp.write('Annotation\t{}\n'.format(gff_path))
    fp.write('//\n')


def main():
    args = parse_args()

    # read index TSV file.
    with open(args.index_refseq, 'r') as in_f:
        with open(args.output_path, 'w') as out_f:
            out_f.write('//\n')
            reader = csv.DictReader(in_f, delimiter='\t')
            for row in reader:
                accession = row['Accession']
                species = row['Species']
                gcf = row['Assembly']
                seq_path = row['SeqPath']
                handle_entry(out_f, accession, species, gcf, seq_path)


if __name__ == "__main__":
    main()
