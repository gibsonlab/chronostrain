import argparse
import random
import csv
from pathlib import Path

import shutil
from multiprocessing.dummy import Pool
from typing import Tuple, Dict, List

from chronostrain import cfg, logger
from chronostrain.database import AbstractStrainDatabase
from chronostrain.util.external.art import art_illumina


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate reads from genomes, using ART.")

    parser.add_argument('-n', '--num_reads', dest='num_reads', required=True, type=int,
                        help='<Required> The number of reads to sample per time point..')
    parser.add_argument('-l', '--read_len', dest='read_len', required=True, type=int,
                        help='<Required> The length of each read.')
    parser.add_argument('-o', '--out_dir', dest='out_dir', required=True, type=str,
                        help='<Required> The directory to output the reads to. The sampler will automatically'
                             'generate a series of fastq files, as well as an index file `input_files.csv`.')
    parser.add_argument('-p', '--profiles', dest='profiles', required=True, nargs=2,
                        help='<Required> A pair of read profiles for paired-end reads. '
                             'The first profile is for the forward strand and the second profile is for the reverse.')
    parser.add_argument('-a', '--abundance_path', dest='abundance_path', required=True, type=str,
                        help='<Required> The path to the abundance CSV file.')

    parser.add_argument('-s', '--seed', dest='seed', required=False, type=int, default=random.randint(0, 100),
                        help='<Optional> The random seed to use for the samplers. Each timepoint will use a unique '
                             'seed, starting with the specified value and incremented by one at a time.')
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
    parser.add_argument('--num_cores', dest='num_cores', required=False, type=int, default=1,
                        help='<Optional> The number of cores to use. If greater than 1, will spawn child '
                             'processes to call art_illumina.')
    parser.add_argument('--clean_after_finish', dest='cleanup', action='store_true',
                        help='<Optional> If flag is turned on, removes all temporary fastq files after execution.')

    return parser.parse_args()


def main():
    args = parse_args()
    strain_db = cfg.database_cfg.get_database()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timepoint_indexed_files = []
    seed = args.seed
    for t, abundance_t in parse_abundance_profile(args.abundance_path):
        out_path_t = out_dir / "reads_{t}.fastq".format(t=t)
        tmpdir = out_dir / "tmp_{t}".format(t=t)
        Path(tmpdir).mkdir(parents=True, exist_ok=True)

        sample_reads_from_rel_abundances(
            final_reads_path=out_path_t,
            abundances=abundance_t,
            num_reads=args.num_reads,
            strain_db=strain_db,
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
        seed = seed + 1
        timepoint_indexed_files.append((t, out_path_t))
    logger.info("Sampled reads to {}".format(args.out_dir))

    index_path = out_dir / "input_files.csv"
    create_index_file(index_path, timepoint_indexed_files)
    logger.info("Wrote index file to {}.".format(index_path))


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
                                     num_reads: int,
                                     strain_db: AbstractStrainDatabase,
                                     tmp_dir: Path,
                                     profile_first: Path,
                                     profile_second: Path,
                                     read_len: int,
                                     quality_shift: int,
                                     quality_shift_2: int,
                                     seed: int,
                                     n_cores: int,
                                     cleanup: bool):
    """
    Loop over each timepoint, and invoke art_illumina on each item. Each instance outputs a separate fastq file,
    so concatenate them at the end.

    :param final_reads_path:
    :param abundances:
    :param num_reads:
    :param strain_db:
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
    if n_cores == 1:
        strain_read_paths = []
        for t_index, (accession, rel_abund) in enumerate(abundances.items()):
            strain = strain_db.get_strain(strain_id=accession)

            output_path = art_illumina(
                reference_path=strain.metadata.file_path,
                num_reads=int(rel_abund * num_reads),
                output_dir=tmp_dir,
                output_prefix="{}_".format(accession),
                profile_first=profile_first,
                profile_second=profile_second,
                quality_shift=quality_shift,
                quality_shift_2=quality_shift_2,
                read_length=read_len,
                seed=seed + t_index
            )

            strain_read_paths.append(output_path)
    elif n_cores > 1:
        configs = [(
            strain_db.get_strain(accession).metadata.file_path,
            int(rel_abund * num_reads),
            tmp_dir,
            "{}_".format(accession),
            profile_first,
            profile_second,
            read_len,
            seed + t_index
        ) for t_index, (accession, rel_abund) in enumerate(abundances.items())]

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
