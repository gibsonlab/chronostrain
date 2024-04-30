import subprocess
from pathlib import Path
from typing import Optional

from .commandline import CommandLineException, call_command


def ssw_align(
        target_path: Path,
        query_path: Path,
        match_score: int,
        mismatch_penalty: int,
        gap_open_penalty: int,
        gap_extend_penalty: int,
        output_path: Path,
        score_threshold: Optional[int] = None,
        best_of_strands: bool = False,
        include_sam_headers: bool = False
):
    """
    ssw-align wrapper
    """
    params = [
        '-m', match_score,
        '-x', mismatch_penalty,
        '-o', gap_open_penalty,
        '-e', gap_extend_penalty,
        '-sc',  # output SAM format (must include backtracing calculation).
        target_path,
        query_path
    ]

    if score_threshold is not None:
        params += ['-f', score_threshold]
    if best_of_strands:
        params.append('-r')
    if include_sam_headers:
        params.append('-h')

    exit_code = call_command(
        command='ssw-align',
        args=params,
        stdout=output_path,
        stderr=subprocess.DEVNULL
    )

    if exit_code != 0:
        raise CommandLineException('ssw-align', exit_code)
