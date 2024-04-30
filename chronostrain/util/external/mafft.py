from pathlib import Path
from typing import Optional, List

from .commandline import CommandLineException, call_command


def mafft_global(
        input_fasta_path: Path,
        output_path: Path,
        n_threads: int = -1,
        auto: bool = False,
        max_iterates: int = 1000
):
    """
    mafft --globalpair --maxiterate 1000 input [> output]
    """
    params = [
        '--nuc',
        '--quiet',
        '--globalpair',
        '--maxiterate', max_iterates
    ]
    params += ['--thread', str(n_threads)]
    if auto:
        params.append('--auto')

    params.append(input_fasta_path)

    exit_code = call_command(
        command='mafft',
        args=params,
        stdout=output_path
    )

    if exit_code != 0:
        raise CommandLineException('mafft', exit_code)


def mafft_add(
        existing_alignment_fasta: Path,
        new_fasta_path: Path,
        output_path: Path,
        n_threads: int = -1,
        reorder: bool = False,
        auto: bool = False,
        mafft_quiet: bool = True,
        keep_length: bool = False,
        silent: bool = False
):
    # Biopython's interface appears outdated (as of 10/23/2021). Use our own cline interface.
    params = [
        '--add', new_fasta_path,
        '--nuc'
    ]
    if mafft_quiet:
        params.append('--quiet')
    if auto:
        params.append('--auto')
    if reorder:
        params.append('--reorder')
    if keep_length:
        params.append('--keeplength')
    params += ['--thread', str(n_threads)]
    params.append(str(existing_alignment_fasta))

    exit_code = call_command(
        command='mafft',
        args=params,
        stdout=output_path,
        silent=silent
    )

    if exit_code != 0:
        raise CommandLineException('mafft', exit_code)


def mafft_fragment(
        reference_fasta_path: Path,
        fragment_fasta_path: Path,
        output_path: Path,
        n_threads: int = -1,
        reorder: bool = False,
        auto: bool = False,
        max_n_frac: float = 0.05,
        gap_open_penalty_group: float = 1.53,
        gap_offset_group: float = 0.0,
        kimura: int = 2,
        pairwise_seeds: Optional[List[Path]] = None,
        quiet: bool = True
):
    # Biopython's interface appears outdated (as of 10/23/2021). Use our own cline interface.
    params = [
        '--addfragments', fragment_fasta_path,
        '--nuc',
        '--maxambiguous', max_n_frac,
        '--op', gap_open_penalty_group,
        '--ep', gap_offset_group,
        '--kimura', kimura
    ]
    if quiet:
        params.append('--quiet')
    if auto:
        params.append('--auto')
    if reorder:
        params.append('--reorder')
    params += ['--thread', str(n_threads)]
    params.append(str(reference_fasta_path))

    if pairwise_seeds is not None:
        for p in pairwise_seeds:
            params += ['--seed', p]

    exit_code = call_command(
        command='mafft',
        args=params,
        stdout=output_path
    )

    if exit_code != 0:
        raise CommandLineException('mafft', exit_code)
