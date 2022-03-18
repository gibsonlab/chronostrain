import os

from typing import Union

from .commandline import *


def make_blast_db(
        input_fasta: Path,
        db_dir: Path,
        db_name: str,
        is_nucleotide: bool,
        title: str,
        parse_seqids: bool
):
    params = [
        '-in', input_fasta,
        '-out', db_name,
        '-dbtype',
        'nucl' if is_nucleotide else 'prot',
        '-title', title,
    ]
    if parse_seqids:
        params += ['-parse_seqids']

    exit_code = call_command(
        command='makeblastdb',
        args=params,
        cwd=db_dir
    )
    if exit_code != 0:
        raise CommandLineException('makeblastdb', exit_code)


def blastn(
        db_name: str,
        query_fasta: Path,
        out_path: Path,
        db_dir: Optional[Path] = None,
        evalue_max: Optional[float] = None,
        perc_identity_cutoff: Optional[int] = None,
        out_fmt: Union[str, int] = 6,  # 6: TSV without comments
        max_target_seqs: Optional[int] = None,
        max_hsps: Optional[int] = None,
        num_threads: Optional[int] = None,
        query_coverage_hsp_percentage: Optional[float] = None,
        strand: str = 'both',
        remote: bool = False,
        entrez_query: Optional[str] = None,
        taxidlist_path: Optional[Path] = None
):
    params = [
        '-db', db_name,
        '-query', query_fasta,
        '-outfmt', f"\"{out_fmt}\"",
        '-out', out_path,
        '-strand', strand
    ]

    if num_threads is not None:
        params += ['-num_threads', num_threads]
    if remote:
        params.append('-remote')
    if entrez_query is not None:
        params += ['-entrez_query', f"\"{entrez_query}\""]
    if perc_identity_cutoff is not None:
        params += ['-perc_identity', perc_identity_cutoff]
    if evalue_max is not None:
        params += ['-evalue', evalue_max]
    if max_target_seqs is not None:
        params += ['-max_target_seqs', max_target_seqs]
    if max_hsps is not None:
        params += ['-max_hsps', max_hsps]
    if query_coverage_hsp_percentage is not None:
        params += ['-qcov_hsp_perc', query_coverage_hsp_percentage]
    if taxidlist_path is not None:
        params += ['-taxidlist', taxidlist_path]

    if db_dir is not None:
        env = os.environ.copy()
        env['BLASTDB'] = str(db_dir)

        exit_code = call_command(
            command='blastn',
            args=params,
            environment=env
        )
    else:
        exit_code = call_command(command='blastn', args=params)
    if exit_code != 0:
        raise CommandLineException('blastn', exit_code)


def tblastn(
        db_name: str,
        db_dir: Path,
        query_fasta: Path,
        evalue_max: float,
        out_path: Path,
        out_fmt: Union[str, int] = 6,  # 6: TSV without comments
        max_target_seqs: Optional[int] = None,
        max_hsps: Optional[int] = None,
        num_threads: int = 1,
        query_coverage_hsp_percentage: Optional[float] = None
):
    params = [
        '-db', db_name,
        '-num_threads', num_threads,
        '-query', query_fasta,
        '-evalue', evalue_max,
        '-outfmt', out_fmt,
        '-out', out_path
    ]

    if max_target_seqs is not None:
        params += ['-max_target_seqs', max_target_seqs]
    if max_hsps is not None:
        params += ['-max_hsps', max_hsps]
    if query_coverage_hsp_percentage is not None:
        params += ['-qcov_hsp_perc', query_coverage_hsp_percentage]

    env = os.environ.copy()
    env['BLASTDB'] = str(db_dir)

    exit_code = call_command(
        command='tblastn',
        args=params,
        environment=env
    )
    if exit_code != 0:
        raise CommandLineException('tblastn', exit_code)
