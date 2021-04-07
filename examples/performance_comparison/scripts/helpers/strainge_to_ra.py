import os
from typing import Tuple, Dict

import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-i', '--input', dest="inputs",
                        required=True, type=str, action='append',
                        help='<Required> The TSV file path pointing to the output of StrainGST. '
                             'Pass argument multiple times to specify multiple files. '
                             'Should appear in the same order as `-t`.')
    parser.add_argument('-t', '--time_point', dest="time_points",
                        required=True, type=float, action='append',
                        help='<Required> Time points corresponding to the input file in the `-i` argument.'
                             'Should appear in the same order corresponding to `-i`.')
    parser.add_argument('-o', '--output_path', dest="output_path",
                        required=True, type=str,
                        help='<Required> The output file path.')

    parser.add_argument('--strain_trim', required=False, type=str,
                        default=".fa.gz",
                        help='<Optional> The substring to trim off of each strain name. '
                             'Useful if inputs were generated while keeping a .fasta (or similar) file extensionm,'
                             'since StrainGST outputs strain names with the file extension still attached.')
    return parser.parse_args()


def parse_indices(header_line: str) -> Tuple[int, int]:
    tokens = header_line.split("\t")
    rapct_idx = -1
    strain_idx = -1
    for i, token in enumerate(tokens):
        if token == "rapct":
            rapct_idx = i
        if token == "strain":
            strain_idx = i
    if rapct_idx == -1:
        raise RuntimeError("Could not find 'rapct' header in file.")
    if strain_idx == -1:
        raise RuntimeError("Could not find 'strain' header in file.")
    return rapct_idx, strain_idx


def extract_strain(strain_token: str, trim: str) -> str:
    return strain_token[:-len(trim)]


def extract_abunds(input_path: str, strain_trim: str) -> Dict[str, float]:
    values = []
    strains = []

    with open(input_path, "r") as input_file:
        # Read the first three lines.
        _ = input_file.readline()
        _ = input_file.readline()
        header_line = input_file.readline()
        rapct_idx, strain_idx = parse_indices(header_line)

        for line in input_file:
            row_tokens = line.split("\t")
            abund_val = float(row_tokens[rapct_idx])
            values.append(abund_val)

            strain_id = extract_strain(row_tokens[strain_idx], trim=strain_trim)
            strains.append(strain_id)

    abundances = np.array(values) / np.sum(values)

    return {
        strain: abnd
        for strain, abnd in zip(strains, abundances)
    }


def main():
    args = parse_args()

    abundances = []
    for t, input_path in zip(args.time_points, args.inputs):
        print("{t}: {i}".format(
            t=t,
            i=input_path
        ))
        abundances.append(extract_abunds(input_path, strain_trim=args.strain_trim))

    strains = set()
    for abund_t in abundances:
        for strain_key in abund_t.keys():
            strains.add(strain_key)
    strains = list(strains)

    with open(args.output_path, "w") as outfile:
        outfile.write('"T"')
        for strain in strains:
            outfile.write(',"{}"'.format(strain))
        outfile.write("\n")

        for t, abund_t in zip(args.time_points, abundances):
            outfile.write('"{}"'.format(t))
            for strain in strains:
                outfile.write(',"{}"'.format(
                    abund_t.get(strain, 0.0)
                ))
            outfile.write("\n")

    print("Wrote output to file {}.".format(args.output_path))


if __name__ == "__main__":
    main()