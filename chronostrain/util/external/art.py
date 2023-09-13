from pathlib import Path
from typing import Optional, Tuple
from .commandline import call_command, CommandLineException


def art_illumina(reference_path: Path,
                 num_reads: int,
                 output_dir: Path,
                 output_prefix: str,
                 profile_first: Path,
                 profile_second: Path,
                 read_length: int,
                 seed: int,
                 paired_end_frag_mean_len: int = 10000,
                 paired_end_frag_stdev_len: int = 200,
                 output_sam: bool = False,
                 output_aln: bool = True,
                 quality_shift: Optional[int] = None,
                 quality_shift_2: Optional[int] = None,
                 stdout_path: Optional[Path] = None,
                 silent: bool = False) -> Tuple[Path, Path]:
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
    :param paired_end_frag_mean_len:
    :param paired_end_frag_stdev_len:
    :param output_sam:
    :param quality_shift:
    :param quality_shift_2:
    :param stdout_path:
    :param silent:
    :return: The filepaths to the synthetic paired-end reads.
    """

    cmd_args = [
        '--qprof1', str(profile_first),
        '--qprof2', str(profile_second),
        '-i', reference_path,
        '-l', read_length,
        '-c', num_reads,
        '-p',
        '-m', paired_end_frag_mean_len,
        '-s', paired_end_frag_stdev_len,
        '-o', output_prefix,
        '-rs', str(seed)
    ]

    if output_sam:
        cmd_args.append('-sam')

    if not output_aln:
        cmd_args.append('-na')

    if isinstance(quality_shift, int):
        cmd_args = cmd_args + ['-qs', str(quality_shift)]
    if isinstance(quality_shift_2, int):
        cmd_args = cmd_args + ['-qs2', str(quality_shift_2)]

    exit_code = call_command(
        'art_illumina',
        args=cmd_args,
        cwd=output_dir,
        stdout=stdout_path,
        silent=silent
    )
    if exit_code != 0:
        raise CommandLineException("art_illumina", exit_code)
    else:
        return output_dir / "{}1.fq".format(output_prefix), output_dir / "{}2.fq".format(output_prefix)
