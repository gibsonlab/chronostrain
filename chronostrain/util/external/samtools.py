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
        output_path: Path,
        fai_reference_path: Path,
        reference_path: Path,
        exclude_unmapped: bool
):
    # Biopython's interface appears outdated (as of 10/23/2021). Use our own cline interface.
    params = [
        'view',
        '-t', fai_reference_path,
        '-T', reference_path,
        '-S',
        sam_path,
        '-b',
        '-o', output_path
    ]
    if exclude_unmapped:
        params += ['-F', 4]
    return _samtools(params)


def sam_filter(
        sam_or_bam_path: Path,
        output_path: Path,
        exclude_unmapped: bool,
        exclude_header: bool
):
    params = [
        'view',
        sam_or_bam_path,
        '-o', output_path,
    ]
    if exclude_unmapped:
        params += ['-F', 4]
    if exclude_header:
        params += ['--no-header']

    return _samtools(params)


def bam_sort(
        sam_or_bam_path: Path,
        output_path: Path,
        num_threads: int = 1
):
    params = [
        'sort',
        sam_or_bam_path,
        '-o', output_path,
        '-@', num_threads
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


def index(
        sam_compressed_or_bam: Path,
):
    return _samtools(
        ['index', sam_compressed_or_bam]
    )
