import os
from typing import Optional
from .commandline import call_command, CommandLineException


def art_illumina(reference_path: str,
                 num_reads: int,
                 output_dir: str,
                 output_prefix: str,
                 profile_first: str,
                 profile_second: str,
                 read_length: int,
                 seed: int,
                 quality_shift: Optional[int] = None,
                 quality_shift_2: Optional[int] = None) -> str:
    """
    Call art_illumina.

    :param reference_path:
    :param num_reads:
    :param output_dir:
    :param output_prefix:
    :param profile_first:
    :param profile_second:
    :param read_length:
    :param seed:
    :param quality_shift:
    :param quality_shift_2:
    :return: The filepath to the paired-end reads. TODO: Currently only returns the first read of the pair.
    """

    cmd_args = ['--qprof1', profile_first,
     '--qprof2', profile_second,
     '-sam',
     '-i', reference_path,
     '-l', str(read_length),
     '-c', str(num_reads),
     '-p',
     '-m', '200',
     '-s', '10',
     '-o', output_prefix,
     '-rs', str(seed)]

    if isinstance(quality_shift, int):
        cmd_args = cmd_args + ['-qs', str(quality_shift)]
    if isinstance(quality_shift_2, int):
        cmd_args = cmd_args + ['-qs2', str(quality_shift_2)]

    exit_code = call_command(
        'art_illumina',
        args=cmd_args,
        cwd=output_dir
    )
    if exit_code != 0:
        raise CommandLineException("art_illumina", exit_code)
    else:
        return os.path.join(output_dir, "{}1.fq".format(output_prefix))
