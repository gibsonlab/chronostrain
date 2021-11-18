from pathlib import Path
from typing import Optional, List

from .commandline import CommandLineException, call_command


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
        pairwise_seeds: Optional[List[Path]] = None,
):
    # Biopython's interface appears outdated (as of 10/23/2021). Use our own cline interface.
    params = [
        '--addfragments', fragment_fasta_path,
        '--nuc',
        '--quiet',
        '--maxambiguous', max_n_frac,
        '--op', gap_open_penalty_group,
        '--ep', gap_offset_group,
    ]
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
        output_path=output_path
    )

    if exit_code != 0:
        raise CommandLineException('mafft', exit_code)
