from pathlib import Path

from Bio.Align.Applications import ClustalOmegaCommandline

from .commandline import CommandLineException, call_command


def clustal_omega(
        input_path: Path,
        output_path: Path,
        force: bool = False,
        verbose: bool = False,
        out_format: str = 'fasta',
        auto: bool = False
):
    cline = ClustalOmegaCommandline(
        infile=input_path,
        outfile=output_path,
        force=force,
        verbose=verbose,
        outfmt=out_format,
        auto=auto
    )

    tokens = str(cline).split()

    exit_code = call_command(
        tokens[0],
        args=tokens[1:]
    )
    if exit_code != 0:
        raise CommandLineException(tokens[0], exit_code)
