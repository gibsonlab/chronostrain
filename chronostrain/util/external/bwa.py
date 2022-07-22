from typing import Union, Tuple

from .commandline import *


def bwa_index(reference_path: Path, bwa_cmd='bwa-mem2'):
    exit_code = call_command(
        bwa_cmd,
        ['index', reference_path]
    )
    if exit_code != 0:
        raise CommandLineException(f"{bwa_cmd} index", exit_code)


def bwa_mem(output_path: Path,
            reference_path: Path,
            read_path: Path,
            min_seed_length: int,
            reseed_ratio: float = 1.5,
            bandwidth: int = 100,
            num_threads: int = 1,
            report_all_alignments: bool = False,
            off_diag_dropoff: int = 100,
            match_score: int = 1,
            mismatch_penalty: int = 4,
            gap_open_penalty: Union[int, Tuple[int, int]] = 6,
            gap_extend_penalty: Union[int, Tuple[int, int]] = 1,
            clip_penalty: int = 5,
            score_threshold: int = 30,
            unpaired_penalty: int = 17,
            soft_clip_for_supplementary: bool = False,
            bwa_cmd='bwa-mem2'):
    params = [
        'mem',
        '-o', output_path,
        '-t', num_threads,
        '-k', str(min_seed_length),
        '-r', reseed_ratio,
        '-w', bandwidth,
        '-d', off_diag_dropoff,
        '-A', match_score,
        '-B', mismatch_penalty,
        '-L', clip_penalty,
        '-T', score_threshold,
        '-U', unpaired_penalty,
        '-v', 2
    ]

    if isinstance(gap_open_penalty, Tuple):
        params += ['-O', f'{gap_open_penalty[0]},{gap_open_penalty[1]}']
    else:
        params += ['-O', gap_open_penalty]

    if isinstance(gap_extend_penalty, Tuple):
        params += ['-E', f'{gap_extend_penalty[0]},{gap_extend_penalty[1]}']
    else:
        params += ['-E', gap_extend_penalty]

    if report_all_alignments:
        params.append('-a')
    if soft_clip_for_supplementary:
        params.append('-Y')

    exit_code = call_command(
        command=bwa_cmd,
        args=params + [reference_path, read_path]
    )
    if exit_code != 0:
        raise CommandLineException(f"{bwa_cmd} mem", exit_code)


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
