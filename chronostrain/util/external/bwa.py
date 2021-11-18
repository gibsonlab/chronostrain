from .commandline import *


def bwa_index(reference_path: Path, bwa_path="bwa", algorithm: str = "is"):
    exit_code = call_command(
        bwa_path,
        [
            'index',
            '-a', algorithm,
            reference_path
        ]
    )
    if exit_code != 0:
        raise CommandLineException("bwa index", exit_code)


def bwa_mem(output_path: Path,
            reference_path: Path,
            read_path: Path,
            min_seed_length: int,
            num_threads: int = 1,
            report_all_alignments: bool = False,
            off_diag_dropoff: int = 100,
            match_score: int = 1,
            mismatch_penalty: int = 4,
            gap_open_penalty: int = 6,
            gap_extend_penalty: int = 1,
            clip_penalty: int = 5,
            bwa_path="bwa"):
    params = [
        'mem',
        '-o', output_path,
        '-t', num_threads,
        '-k', str(min_seed_length),
        '-d', off_diag_dropoff,
        '-A', match_score,
        '-B', mismatch_penalty,
        '-O', gap_open_penalty,
        '-E', gap_extend_penalty,
        '-L', clip_penalty,
        reference_path,
        read_path
    ]
    if report_all_alignments:
        params.insert(5, '-a')
    exit_code = call_command(
        command=bwa_path,
        args=params
    )
    if exit_code != 0:
        raise CommandLineException("bwa mem", exit_code)
