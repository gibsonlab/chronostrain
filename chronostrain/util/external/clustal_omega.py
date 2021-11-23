from pathlib import Path
from typing import Optional

from .commandline import CommandLineException, call_command


def clustal_omega(
        output_path: Path,
        input_path: Optional[Path] = None,
        profile1: Optional[Path] = None,
        profile2: Optional[Path] = None,
        force_overwrite: bool = False,
        verbose: bool = False,
        out_format: str = 'fasta',
        auto: bool = False,
        seqtype: str = 'DNA',
        guidetree_out: Optional[Path] = None,
        guidetree_in: Optional[Path] = None,
        silent: bool = False,
        n_threads: int = 1
):
    params = [
        '-o', output_path,
        f'--outfmt={out_format}',
        '-t', seqtype
    ]

    if input_path is not None:
        params += ['-i', input_path]
    if profile1 is not None:
        params += ['--profile1', profile1]
    if profile2 is not None:
        params += ['--profile2', profile2]

    if auto:
        params.append('--auto')
    if force_overwrite:
        params.append('--force')
    if verbose:
        params.append('--verbose')

    if guidetree_out is not None:
        params.append(f'--guidetree-out={str(guidetree_out)}')

    if guidetree_in is not None:
        params.append(f'--guidetree-in={str(guidetree_in)}')

    if n_threads > 1:
        params.append(f'--threads={n_threads}')

    exit_code = call_command(
        command='clustalo',
        args=params,
        silent=silent
    )
    if exit_code != 0:
        raise CommandLineException('clustalo', exit_code)
