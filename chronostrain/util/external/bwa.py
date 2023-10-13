from .commandline import *


def bwa_index(reference_path: Path, bwa_cmd='bwa-mem2', check_suffix: str = None):
    if check_suffix is not None:
        if reference_path.with_suffix(f'{reference_path.suffix}.{check_suffix}').exists():
            return

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
            mem_discard_threshold: int = 10000,  ## -c, only in newer BWA versions
            chain_drop_threshold: float = 0.5,  # -D, only in newer BWA versions
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
        '-t', num_threads,
        '-k', str(min_seed_length),
        '-r', reseed_ratio,
        '-c', mem_discard_threshold,  # only newer bwa versions
        '-D', chain_drop_threshold,  # only newer bwa versions
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

    # if exclude_unmapped:
    #     exit_code = call_command(
    #         command=bwa_cmd,
    #         args=params + [reference_path, read_path],
    #         piped_command=f'samtools view -F 4 -o {output_path}',
    #         shell=True
    #     )
    #     if exit_code != 0:
    #         raise CommandLineException(f"{bwa_cmd} mem", exit_code)
    # else:
    exit_code = call_command(
        command=bwa_cmd,
        args=params + ['-o', output_path] + [reference_path, read_path]
    )
    if exit_code != 0:
        raise CommandLineException(f"{bwa_cmd} mem", exit_code)


def bwa_fastmap(output_path: Path,
                reference_path: Path,
                query_path: Path,
                min_smem_len: Optional[int] = None,
                max_interval_size: int = 20,
                silent: bool = True):
    args = ['fastmap',
            reference_path,
            query_path,
            '-w', max_interval_size]
    if min_smem_len is not None and isinstance(min_smem_len, int):
        args += ['-l', min_smem_len]

    exit_code = call_command(
        command='bwa',
        args=args,
        stdout=output_path,
        silent=silent
    )
    if exit_code != 0:
        raise CommandLineException("bwa fastmap", exit_code)
