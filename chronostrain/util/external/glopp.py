from typing import Optional
from pathlib import Path

from chronostrain.config import cfg
from .commandline import CommandLineException, call_command


def run_glopp(
        bam_path: Path,
        vcf_path: Path,
        output_dir: Path,
        use_mec_score: bool = False,
        ploidy: Optional[int] = None
):
    params = [
        '-b', str(bam_path),
        '-c', str(vcf_path),
        '-o', str(output_dir)
    ]
    if ploidy is not None:
        params += ['-p', str(ploidy)]
    if use_mec_score:
        params.append("-m")

    exit_code = call_command(
        command=cfg.external_tools_cfg.glopp_path,
        args=params
    )

    if exit_code != 0:
        raise CommandLineException('glopp', exit_code)


def run_flopp(
        bam_path: Path,
        vcf_path: Path,
        output_path: Path,
        ploidy: int
):
    params = [
        '-b', str(bam_path),
        '-c', str(vcf_path),
        '-o', str(output_path),
        '-p', str(ploidy)
    ]

    exit_code = call_command(
        command=cfg.external_tools_cfg.flopp_path,
        args=params
    )

    if exit_code != 0:
        raise CommandLineException('flopp', exit_code)
