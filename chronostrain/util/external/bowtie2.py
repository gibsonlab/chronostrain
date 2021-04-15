from pathlib import Path
from typing import List

from .commandline import call_command, CommandLineException


def bowtie2_inspect(basename: str,
                    out_path: Path = None,
                    command_path: str = "bowtie2-inspect"):
    """
    :param basename: bt2 filename minus trailing .1.bt2/.2.bt2
    :param out_path: The file to save the output to. Equivalent to `bowtie2-inspect [basename] > [out_path]`.
    :param command_path: The full path to `bowtie2-inspect`, if not located in path env (typically /usr/bin).
    :return:
    """
    exit_code = call_command(
        command_path,
        args=[basename],
        output_path=out_path
    )
    if exit_code != 0:
        raise CommandLineException("bowtie2-inspect", exit_code)


def bowtie2_build(refs_in: List[Path],
                  output_index_basename: str,
                  out_path: Path,
                  command_path: str = "bowtie2-build"):
    """
    :param refs_in: List of paths to reference sequences.
    :param output_index_basename: write bt2 data to files with this dir/basename.
    :param command_path: The path to `bowtie2-inspect`, if not located in path env (typically /usr/bin).
    :return:
    """
    exit_code = call_command(
        command_path,
        args=[",".join(str(p for p in refs_in)), output_index_basename],
        output_path=out_path
    )
    if exit_code != 0:
        raise CommandLineException("bowtie2-build", exit_code)
