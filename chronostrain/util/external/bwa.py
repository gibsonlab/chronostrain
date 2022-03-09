from .commandline import *


def bwa_index(reference_path: Path, algorithm: str = "is"):
    exit_code = call_command(
        'bwa',
        ['index', '-a', algorithm, reference_path]
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
            clip_penalty: int = 5):
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
        command='bwa',
        args=params
    )
    if exit_code != 0:
        raise CommandLineException("bwa mem", exit_code)


def bwa_fastmap(output_path: Path,
                reference_path: Path,
                query_path: Path,
                min_smem_len: Optional[int] = None,
                max_interval_size: int = 20):
    args = ['fastmap',
            reference_path,
            query_path,
            '-w', max_interval_size]
    if min_smem_len is not None and isinstance(min_smem_len, int):
        args += ['-l', min_smem_len]

    exit_code = call_command(
        command='bwa',
        args=args,
        output_path=output_path
    )
    if exit_code != 0:
        raise CommandLineException("bwa fastmap", exit_code)
