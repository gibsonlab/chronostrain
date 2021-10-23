from pathlib import Path

from .commandline import CommandLineException, call_command


def mafft_fragment(
        reference_fasta_path: Path,
        fragment_fasta_path: Path,
        output_path: Path,
        n_threads: int = -1,
        reorder: bool = False,
        auto: bool = False
):
    # Biopython's interface appears outdated (as of 10/23/2021). Use our own cline interface.
    params = [
        '--addfragments',
        str(fragment_fasta_path)
    ]
    if auto:
        params.append('--auto')
    if reorder:
        params.append('--reorder')
    params += ['--thread', str(n_threads)]
    params.append(str(reference_fasta_path))

    exit_code = call_command(
        command='mafft',
        args=params,
        output_path=output_path
    )

    if exit_code != 0:
        raise CommandLineException('mafft', exit_code)
