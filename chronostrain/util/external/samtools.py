from pathlib import Path
from typing import Any, List, Optional, TextIO

from .commandline import CommandLineException, call_command


def _samtools(params: List[Any], silent: bool = False, stdout: Optional[TextIO] = None):
    exit_code = call_command(
        command='samtools',
        args=params,
        silent=silent,
        stdout=stdout
    )

    if exit_code != 0:
        raise CommandLineException('samtools', exit_code)


def sam_to_bam(
        sam_path: Path,
        output_path: Path
):
    # Biopython's interface appears outdated (as of 10/23/2021). Use our own cline interface.
    params = [
        'view',
        '-S',
        '-b',
        sam_path,
        '-o', output_path
    ]

    return _samtools(params)


def bam_sort(
        bam_path: Path,
        output_path: Path
):
    params = [
        'sort',
        bam_path,
        '-o', output_path
    ]

    return _samtools(params)


def merge(
        bam_paths: List[Path],
        out_path: Path
):
    params = [
        'merge'
        '-o', out_path
    ]

    return _samtools(params + bam_paths)


def sam_mapped_only(sam_path: Path, output_path: Path):
    params = [
        'view',
        '-F', 4,
        sam_path,
        '-o', output_path
    ]

    return _samtools(params)


def faidx(
        fasta_path: Path,
        query_regions: Optional[List[str]] = None,
        buf: Optional[TextIO] = None,
        silent: bool = False
):
    params = ['faidx']
    params.append(fasta_path)
    if query_regions is not None and len(query_regions) > 0:
        params += query_regions
    return _samtools(params, stdout=buf, silent=silent)
