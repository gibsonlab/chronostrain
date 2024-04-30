from pathlib import Path
from typing import Optional, TextIO

from .commandline import CommandLineException, call_command


def cdbfasta(
        fasta_path: Path,
        silent: bool = False,
) -> Path:
    """
    Invokes cdbfasta and returns the newly created index path.
    """
    params = [fasta_path]

    exit_code = call_command(
        command='cdbfasta',
        args=params,
        silent=silent
    )

    if exit_code != 0:
        raise CommandLineException('cdbfasta', exit_code)

    return fasta_path.with_name(f'{fasta_path.name}.cidx')


def cdbyank(
        index_path: Path,
        target_accession: str,
        buf: Optional[TextIO] = None,
        silent: bool = False
):
    exit_code = call_command(
        command='cdbyank',
        args=['-a', target_accession, index_path],
        silent=silent,
        stdout=buf
    )

    if exit_code != 0:
        raise CommandLineException('cdbyank', exit_code)
