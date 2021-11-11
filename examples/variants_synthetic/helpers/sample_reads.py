import argparse
import random
import csv
from pathlib import Path

import shutil
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

    timepoint_indexed_files = []
    seed = Seed(args.seed)
    for t, abundance_t in parse_abundance_profile(args.abundance_path):
        # Sample a random multinomial profile.
        np.random.seed(seed.next_value())
        read_counts_t = sample_read_counts(args.num_reads, abundance_t)

        # Generate the read path.
        out_path_t = out_dir / "reads_{t}.fastq".format(t=t)
        tmpdir = out_dir / "tmp_{t}".format(t=t)
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

        # Invoke art sampler on each time point.
        sample_reads_from_rel_abundances(
            final_reads_path=out_path_t,
            abundances=read_counts_t,
            strain_paths=parse_strain_paths(Path(args.fasta_dir)),
            tmp_dir=tmpdir,
            profile_first=args.profiles[0],
            profile_second=args.profiles[1],
            read_len=args.read_len,
            quality_shift=args.quality_shift,
            quality_shift_2=args.quality_shift_2,
            seed=seed,
            n_cores=args.num_cores,
            cleanup=args.cleanup
        )
        timepoint_indexed_files.append((t, out_path_t))
    logger.info("Sampled reads to {}".format(args.out_dir))

    index_path = out_dir / "input_files.csv"
    create_index_file(index_path, timepoint_indexed_files)
    logger.info("Wrote index file to {}.".format(index_path))


def parse_strain_paths(fasta_dir: Path) -> Dict[str, Path]:
    assert fasta_dir.is_dir()
    return {
        child.stem: child
        for child in fasta_dir.iterdir()
        if child.is_file() and child.suffix == ".fasta"
    }


def create_index_file(index_path: Path, read_paths: List[Tuple[float, Path]]):
    with open(index_path, 'w') as index_file:
        for time_point, reads_path_t in read_paths:
            print("\"{t}\",\"{file}\"".format(
                t=time_point,
                file=reads_path_t
            ), file=index_file)


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


def sample_reads_from_rel_abundances(final_reads_path: Path,
                                     abundances: Dict[str, float],
                                     strain_paths: Dict[str, Path],
                                     tmp_dir: Path,
                                     profile_first: Path,
                                     profile_second: Path,
                                     read_len: int,
                                     quality_shift: int,
                                     quality_shift_2: int,
                                     seed: Seed,
                                     n_cores: int,
                                     cleanup: bool):
    """
    Loop over each timepoint, and invoke art_illumina on each item. Each instance outputs a separate fastq file,
    so concatenate them at the end.

    :param final_reads_path:
    :param abundances:
    :param tmp_dir:
    :param profile_first:
    :param profile_second:
    :param read_len:
    :param quality_shift:
    :param quality_shift_2:
    :param seed:
    :param n_cores:
    :param cleanup:
    :return:
    """
    for strain_id, _ in abundances.items():
        if not strain_id in strain_paths:
            raise ValueError(
                f"Abundances file requests reads for `{strain_id}`, but couldn't find corresponding fasta file."
            )

    if n_cores == 1:
        strain_read_paths = []
        for entry_index, (strain_id, read_count) in enumerate(abundances.items()):
            fasta_path = strain_paths[strain_id]

            output_path = art_illumina(
                reference_path=fasta_path,
                num_reads=read_count,
                output_dir=tmp_dir,
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

            strain_read_paths.append(output_path)
    elif n_cores > 1:
        configs = [(
            strain_paths[strain_id],
            read_count,
            tmp_dir,
            "{}_".format(strain_id),
            profile_first,
            profile_second,
            read_len,
            seed.next_value()
        ) for entry_index, (strain_id, read_count) in enumerate(abundances.items())]

        thread_pool = Pool(n_cores)
        strain_read_paths = thread_pool.starmap(art_illumina, configs)
    else:
        raise ValueError("# cores must be positive. Got: {}".format(n_cores))

    # Concatenate all results into single file.
    logger.debug("Concatenating {} read files to {}.".format(len(strain_read_paths), final_reads_path))
    concatenate_files(strain_read_paths, final_reads_path)

    if cleanup:
        logger.debug("Cleaning up temp directory {}.".format(tmp_dir))
        shutil.rmtree(tmp_dir)


def concatenate_files(input_paths: List[Path], output_path: Path):
    """
    Concatenates the contents of each file in input_paths into output_path.
    Identical to cat (*) > output_path in a for loop.
    :param input_paths:
    :param output_path:
    :return:
    """
    with open(output_path, "w") as out_file:
        for i, in_path in enumerate(input_paths):
            logger.debug("File {} of {}. [{}]".format(
                i+1,
                len(input_paths),
                in_path
            ))
            with open(in_path, "r") as in_file:
                shutil.copyfileobj(in_file, out_file)


if __name__ == "__main__":
    main()
