import argparse
import random
import csv
from pathlib import Path

from multiprocessing.dummy import Pool
from typing import Tuple, Dict, List

from chronostrain import logger
from chronostrain.util.external.art import art_illumina

from random import seed, randint
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate reads from specified variant genomes, using ART.")

    # ============== Required params
    parser.add_argument('-f', '--fasta_dir', required=True, type=str,
                        help='<Required> The directory which contains all of the variant genome FASTA files.')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='<Required> The directory to which the reads should be output to.')
    parser.add_argument('-a', '--abundance_path', dest='abundance_path', required=True, type=str,
                        help='<Required> The path to the abundance CSV file.')
    parser.add_argument('-n', '--num_reads', dest='num_reads', required=True, type=int,
                        help='<Required> The number of reads to sample per time point..')
    parser.add_argument('-p', '--profiles', dest='profiles', required=True, nargs=2,
                        help='<Required> A pair of read profiles for paired-end reads. '
                             'The first profile is for the forward strand and the second profile is for the reverse.')
    parser.add_argument('-l', '--read_len', dest='read_len', required=True, type=int,
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
    parser.add_argument('--clean_after_finish', dest='cleanup', action='store_true',
                        help='<Optional> If flag is turned on, removes all temporary fastq files after execution.')

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

    # Invoke art sampler on each time point.
    sample_reads_from_rel_abundances(
        time_points=time_points,
        read_counts=read_counts,
        strain_paths=parse_strain_paths(Path(args.fasta_dir)),
        out_dir=out_dir,
        profile_first=args.profiles[0],
        profile_second=args.profiles[1],
        read_len=args.read_len,
        quality_shift=args.quality_shift,
        quality_shift_2=args.quality_shift_2,
        seed=master_seed,
        n_cores=args.num_cores,
        index_path=index_path
    )

    logger.info("Sampled reads to {}".format(args.out_dir))


def parse_strain_paths(fasta_dir: Path) -> Dict[str, Path]:
    assert fasta_dir.is_dir()
    return {
        child.stem: child
        for child in fasta_dir.iterdir()
        if child.is_file() and child.suffix == ".fasta"
    }


def create_index_file(index_path: Path, entries: List[Tuple[float, int, Path]]):
    with open(index_path, 'w') as index_file:
        for time_point, n_reads, reads_path_t in entries:
            print(f'"{time_point}","{n_reads}","{str(reads_path_t)}"', file=index_file)


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
        time_points: List[float],
        read_counts: List[Dict[str, int]],
        strain_paths: Dict[str, Path],
        out_dir: Path,
        profile_first: Path,
        profile_second: Path,
        read_len: int,
        quality_shift: int,
        quality_shift_2: int,
        seed: Seed,
        n_cores: int,
        index_path: Path
):
    """
    Loop over each timepoint, and invoke art_illumina on each item. Each instance outputs a separate fastq file,
    so concatenate them at the end.
    """
    for strain_dict in read_counts:
        for strain_id, _ in strain_dict.items():
            if strain_id not in strain_paths:
                raise ValueError(
                    f"Abundances file requests reads for `{strain_id}`, but couldn't find corresponding fasta file."
                )

    if n_cores == 1:
        index_entries = []
        for time_point, read_counts_t in zip(time_points, read_counts):
            for strain_id, n_reads in read_counts_t.items():
                fasta_path = strain_paths[strain_id]

                output_path = art_illumina(
                    reference_path=fasta_path,
                    num_reads=n_reads,
                    output_dir=out_dir,
                    output_prefix="{}_".format(strain_id),
                    profile_first=profile_first,
                    profile_second=profile_second,
                    quality_shift=quality_shift,
                    quality_shift_2=quality_shift_2,
                    read_length=read_len,
                    seed=seed.next_value(),
                    output_sam=False,
                    output_aln=False
                )

                index_entries.append(
                    (time_point, n_reads, output_path)
                )
    elif n_cores > 1:
        partial_index_entries = []
        configs = []
        for time_point, read_counts_t in zip(time_points, read_counts):
            for strain_id, n_reads in read_counts_t.items():
                fasta_path = strain_paths[strain_id]
                configs.append((
                    fasta_path,
                    n_reads,
                    out_dir,
                    "{}_".format(strain_id),
                    profile_first,
                    profile_second,
                    quality_shift,
                    quality_shift_2,
                    read_len,
                    seed.next_value(),
                    False,
                    False
                ))

                partial_index_entries.append(
                    (time_point, n_reads)
                )

        with Pool(n_cores) as pool:
            result_files = pool.starmap(art_illumina, configs)
            index_entries = []
            for (time_point, n_reads), result_file in zip(partial_index_entries, result_files):
                index_entries.append(
                    (time_point, n_reads, result_file)
                )
    else:
        raise ValueError("# cores must be positive. Got: {}".format(n_cores))

    create_index_file(index_path, index_entries)


if __name__ == "__main__":
    main()
