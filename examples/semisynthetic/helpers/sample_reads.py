import argparse
import random
import csv
from itertools import repeat
from pathlib import Path

import pandas as pd
from multiprocessing.dummy import Pool
from typing import Tuple, Dict, List

from chronostrain.util.external.art import art_illumina
from random import seed, randint
import numpy as np

from chronostrain.logging import create_logger
logger = create_logger("chronostrain.sample_reads")


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate reads from specified variant genomes, using ART.")

    # ============== Required params
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The directory to which the reads should be output to.')
    parser.add_argument('-a', '--abundance_path', dest='abundance_path', required=True, type=str,
                        help='<Required> The path to the abundance CSV file.')
    parser.add_argument('-g', '--genome_dir', dest='genome_dir', type=str,
                        help='<Required> The directory contaiing the genomes to simulate reads from.')
    parser.add_argument('-n', '--num_reads', dest='num_reads', required=True, type=int,
                        help='<Required> The number of synthetic reads to sample per time point.')
    parser.add_argument('-p', '--profiles', dest='profiles', required=True, nargs=2,
                        help='<Required> A pair of read profiles for paired-end reads. '
                             'The first profile is for the forward strand and the second profile is for the reverse.')
    parser.add_argument('-l', '--read_len', dest='read_len', required=False, type=int, default=150,
                        help='<Required> The length of each read.')

    # ============ Optional params
    parser.add_argument('-qs', '--qShift', dest='quality_shift', required=False,
                        type=int, default=None,
                        help='<Optional> The `qShift` argument to pass to art_illumina, which lowers quality scores '
                             'of each read by the specified amount.'
                             '(From the ART documentation: "NOTE: If shifting scores by x, the error rate will '
                             'be 1/(10^(x/10)) of the default profile.")')
    parser.add_argument('-qs2', '--qShift2', dest='quality_shift_2', required=False,
                        type=int, default=None,
                        help='<Optional> (Assuming paired-end reads) The `qShift2` argument to pass to art_illumina, '
                             'which lowers quality scores of each second (reverse half) read by the specified amount.')
    parser.add_argument('-s', '--seed', dest='seed', required=False, type=int, default=random.randint(0, 100),
                        help='<Optional> The random seed to use for the samplers. Each timepoint will use a unique '
                             'seed, starting with the specified value and incremented by one at a time.')
    parser.add_argument('--num_cores', dest='num_cores', required=False, type=int, default=1,
                        help='<Optional> The number of cores to use. If greater than 1, will spawn child '
                             'processes to call art_illumina.')

    return parser.parse_args()


class Seed(object):
    def __init__(self, init: int = 0, min_value: int = 0, max_value: int = 1000000):
        self.value = init
        self.min_value = min_value
        self.max_value = max_value
        seed(self.value)

    def next_value(self):
        r = randint(self.min_value, self.max_value)
        return r


def sample_read_counts(n_reads: int, rel_abund: Dict[str, float]) -> Dict[str, int]:
    """
    :param n_reads:
    :param rel_abund: A dictionary mapping strain IDs to its relative abundance fraction.
    :return: A dictionary mapping strain IDS to read counts, sampled as a multinomial.
    """
    strains = list(rel_abund.keys())
    counts = np.random.multinomial(n=n_reads, pvals=[rel_abund[strain] for strain in strains])
    return {
        strains[i]: counts[i]
        for i in range(len(strains))
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    time_points = []
    read_counts = []

    master_seed = Seed(args.seed)
    for t, abundance_t in parse_abundance_profile(args.abundance_path):
        time_points.append(t)

        # Sample a random multinomial profile.
        np.random.seed(master_seed.next_value())
        read_counts_t = sample_read_counts(args.num_reads, abundance_t)
        read_counts.append(read_counts_t)

    index_path = out_dir / "input_files.csv"
    logger.info(f"Reads will be sampled to {index_path}.")

    # Invoke art sampler on each time point.
    index_entries = sample_reads_from_rel_abundances(
        genome_dir=Path(args.genome_dir),
        time_points=time_points,
        read_counts=read_counts,
        out_dir=out_dir,
        sequencing_sys='HS25',
        read_len=args.read_len,
        quality_shift=args.quality_shift,
        quality_shift_2=args.quality_shift_2,
        insert_len_mean=1000,
        insert_len_std=500,
        seed=master_seed,
        n_cores=args.num_cores
    )

    concatenate_files(time_points, index_entries, out_dir)
    logger.info("Sampled reads to {}".format(out_dir))


def concatenate_files(time_points: List[float], index_entries: List[Tuple[float, int, Path, Path]], out_dir: Path):
    for t_idx, t in enumerate(time_points):
        reads1_all = out_dir / f"{t_idx}_sim_1.fq"
        reads2_all = out_dir / f"{t_idx}_sim_2.fq"

        with open(reads1_all, 'wt') as out_r1, open(reads2_all, 'wt') as out_r2:
            for entry in index_entries:
                if entry[0] != t:
                    continue

                reads1 = entry[2]
                reads2 = entry[3]
                if not reads1.exists():
                    raise Exception(f"Read file {reads1} does not exist. Did ART run correctly?")
                if not reads2.exists():
                    raise Exception(f"Read file {reads2} does not exist. Did ART run correctly?")
                with open(reads1, 'rt') as in_r1:
                    for line in in_r1:
                        out_r1.write(line)
                with open(reads2, 'rt') as in_r2:
                    for line in in_r2:
                        out_r2.write(line)
                reads1.unlink()
                reads2.unlink()


# def create_index_file(time_points: List[float], index_path: Path, entries: List[Tuple[float, int, Path, Path]]):
#     with open(index_path, 'w') as index_file:
#         for t_idx, t in enumerate(time_points):
#             for entry in entries:
#                 if entry[0] != t:
#                     continue
#
#                 n_reads = entry[1]
#                 reads1 = entry[2]
#                 reads2 = entry[3]
#                 print(f'{t},{n_reads},{reads1},paired_1,fastq', file=index_file)
#                 print(f'{t},{n_reads},{reads2},paired_2,fastq', file=index_file)


def parse_abundance_profile(abundance_path: str) -> List[Tuple[float, Dict]]:
    """
    :param abundance_path:
    :return: Output a list of tuples of the following format:
        (time_point, {accession_1: rel_abund_1, ..., accession_N: rel_abund_N})
    """
    with open(abundance_path, 'r') as f:
        abundances = []
        reader = csv.reader(f, delimiter=',', quotechar='"')
        accessions = next(reader)[1:]
        for row in reader:
            t = float(row[0].strip())
            rel_abundances = {
                accession: float(rel_abund_str.strip())
                for accession, rel_abund_str in zip(accessions, row[1:])
            }
            abundances.append((t, rel_abundances))
        return abundances


def sample_reads_from_rel_abundances(
        genome_dir: Path,
        time_points: List[float],
        read_counts: List[Dict[str, int]],
        out_dir: Path,
        sequencing_sys: str,
        read_len: int,
        quality_shift: int,
        quality_shift_2: int,
        seed: Seed,
        insert_len_mean: int,
        insert_len_std: int,
        n_cores: int
) -> List[Tuple[float, int, Path, Path]]:
    """
    Loop over each timepoint, and invoke art_illumina on each item. Each instance outputs a separate fastq file,
    so concatenate them at the end.
    """
    if n_cores == 1:
        index_entries = []
        for t_idx, (time_point, read_counts_t) in enumerate(zip(time_points, read_counts)):
            for strain_id, n_reads in read_counts_t.items():
                fasta_path = genome_dir / f'{strain_id}.fasta'
                if not fasta_path.exists():
                    logger.error(f"Fasta file {fasta_path} does not exist!")
                    exit(1)
                output_path_1, out_path_2 = art_illumina(
                    reference_path=fasta_path,
                    num_reads=n_reads,
                    output_dir=out_dir,
                    output_prefix="{}_{}_".format(t_idx, strain_id),
                    sequencing_sys=sequencing_sys,
                    quality_shift=quality_shift,
                    quality_shift_2=quality_shift_2,
                    read_length=read_len,
                    seed=seed.next_value(),
                    paired_end_frag_mean_len=insert_len_mean,
                    paired_end_frag_stdev_len=insert_len_std,
                    output_sam=False,
                    output_aln=False,
                    stdout_path=out_dir / "{}_{}.out.txt".format(t_idx, strain_id)
                )

                index_entries.append(
                    (time_point, n_reads, output_path_1, out_path_2)
                )
    elif n_cores > 1:
        partial_index_entries = []
        kwargs_iter = []
        for t_idx, (time_point, read_counts_t) in enumerate(zip(time_points, read_counts)):
            for strain_id, n_reads in read_counts_t.items():
                fasta_path = genome_dir / f'{strain_id}.fasta'
                if not fasta_path.exists():
                    logger.error(f"Fasta file {fasta_path} does not exist!")
                    exit(1)
                kwargs = {
                    'reference_path': fasta_path,
                    'num_reads': n_reads,
                    'output_dir': out_dir,
                    'output_prefix': "{}_{}_".format(t_idx, strain_id),
                    'sequencing_sys': sequencing_sys,
                    'quality_shift': quality_shift,
                    'quality_shift_2': quality_shift_2,
                    'read_length': read_len,
                    'seed': seed.next_value(),
                    'paired_end_frag_mean_len': insert_len_mean,
                    'paired_end_frag_stdev_len': insert_len_std,
                    'output_sam': False,
                    'output_aln': False,
                    'stdout_path': out_dir / "{}_{}.out.txt".format(t_idx, strain_id)
                }
                kwargs_iter.append(kwargs)

                partial_index_entries.append(
                    (time_point, n_reads)
                )

        with Pool(n_cores) as pool:
            result_files = starmap_with_kwargs(pool, art_illumina, kwargs_iter)
            index_entries = []
            for (time_point, n_reads), (reads1, reads2) in zip(partial_index_entries, result_files):
                index_entries.append(
                    (time_point, n_reads, reads1, reads2)
                )
    else:
        raise ValueError("# cores must be positive. Got: {}".format(n_cores))

    return index_entries


def starmap_with_kwargs(pool, fn, kwargs_iter):
    kwargs_for_starmap = zip(repeat(fn), kwargs_iter)
    return pool.starmap(apply_kwargs, kwargs_for_starmap)

def apply_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == "__main__":
    main()
