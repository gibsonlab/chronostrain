from pathlib import Path

from .commandline import CommandLineException, call_command


def sam_to_bam(
        sam_path: Path,
        output_path: Path,
):
    # Biopython's interface appears outdated (as of 10/23/2021). Use our own cline interface.
    params = [
        'view', '-S', '-b', str(sam_path)
    ]

    exit_code = call_command(
        command='samtools',
        args=params,
        output_path=output_path
    )

    if exit_code != 0:
        raise CommandLineException('samtools', exit_code)
