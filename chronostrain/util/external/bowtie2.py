import os
from pathlib import Path
from typing import List, Optional, Tuple

from chronostrain.logging import create_logger
from .commandline import call_command, CommandLineException

logger = create_logger(__name__)


def bowtie2_inspect(basename: str,
                    out_path: Path = None,
                    command_path: str = "bowtie2-inspect"):
    """
    :param basename: bt2 filename minus trailing .1.bt2/.2.bt2
    :param out_path: The file to save the output to. Equivalent to `bowtie2-inspect [basename] > [out_path]`.
    :param command_path: The full path to `bowtie2-inspect`, if not located in path env (typically /usr/bin).
    :return:
    """
    exit_code = call_command(
        command_path,
        args=[basename],
        stdout=out_path
    )
    if exit_code != 0:
        raise CommandLineException("bowtie2-inspect", exit_code)


def bowtie2_build(refs_in: List[Path],
                  index_basepath: Path,
                  index_basename: str,
                  bmax: Optional[int] = None,
                  bmaxdivn: Optional[int] = None,
                  offrate: Optional[int] = None,
                  ftabchars: Optional[int] = None,
                  diff_cover_sample: Optional[int] = None,
                  packed: Optional[bool] = False,
                  threads: Optional[int] = None,
                  seed: Optional[int] = None,
                  quiet: bool = False,
                  command_path: str = "bowtie2-build"):
    """
    :param refs_in: List of paths to reference sequences.
    :param index_basepath: The path to which the index is stored.
    :param index_basename: write bt2 data to files with this basename.
    :param bmax: Passed to '--bmax' param.
    :param bmaxdivn: Passed to '--bmaxdivn' param.
    :param diff_cover_sample: Passed to '--dcv' param.
    :param packed: If true, passes the '-p/--packed' flag.
    :param seed: Passed to '--seed' param (the seed to run the command with).
    :param command_path: The path to `bowtie2-inspect`, if not located in path env (typically /usr/bin).
    :param quiet: Suppress debug messages.
    :return:
    """
    args = []

    auto = True
    for optional_param in [bmax, bmaxdivn, diff_cover_sample]:
        if optional_param is not None:
            auto = False

    if packed:
        auto = False

    if not auto:
        args.append('-a')
        if bmax is not None:
            args += ['--bmax', bmax]
        if bmaxdivn is not None:
            args += ['--bmaxdivn', bmaxdivn]
        if diff_cover_sample is not None:
            args += ['--dcv', diff_cover_sample]
        if packed:
            args += ['-p']

    if seed is not None:
        args += ['--seed', seed]
    if offrate is not None:
        args += ['--offrate', offrate]
    if ftabchars is not None:
        args += ['--ftabchars', ftabchars]
    if threads is not None:
        args += ['--threads', threads]
    if quiet:
        args.append('--quiet')

    args += [",".join(str(p) for p in refs_in), index_basepath / index_basename]

    exit_code = call_command(
        command_path,
        args=args,
    )
    if exit_code != 0:
        raise CommandLineException("bowtie2-build", exit_code)


def bt2_func_constant(const: float) -> str:
    return f"C,{const},0"


def bt2_func_linear(const: float, coef: float) -> str:
    return f"L,{const},{coef}"


def bt2_func_sqrt(const: float, coef: float) -> str:
    return f"S,{const},{coef}"


def bt2_func_log(const: float, coef: float) -> str:
    return f"G,{const},{coef}"


def bowtie2(
        index_basepath: Path,
        index_basename: str,
        unpaired_reads: Path,
        out_path: Path,
        aln_seed_num_mismatches: int = 0,
        aln_seed_len: int = 20,
        aln_seed_interval_fn: str = bt2_func_sqrt(1, 1.15),
        aln_gbar: int = 4,
        aln_dpad: int = 15,
        aln_n_ceil: str = bt2_func_linear(0, 0.15),
        score_match_bonus: Optional[int] = None,
        score_min_fn: str = bt2_func_linear(-0.6, -0.6),
        score_mismatch_penalty: Tuple[int, int] = (6, 2),
        score_read_gap_penalty: Tuple[int, int] = (5, 3),
        score_ref_gap_penalty: Tuple[int, int] = (5, 3),
        effort_seed_ext_failures: int = 15,
        effort_num_reseeds: int = 2,
        quality_format: str = 'phred33',
        report_all_alignments: bool = False,
        report_k_alignments: Optional[int] = None,
        num_threads: int = 1,
        rng_seed: int = 0,
        offrate: int = None,
        sam_suppress_noalign: bool = False,
        command_path: str = "bowtie2",
        local: bool = False
):
    if score_mismatch_penalty[0] < score_mismatch_penalty[1]:
        raise ValueError("Score mismatch penalty's MAX must be greater than MIN. (got: {}, {})".format(
            score_mismatch_penalty[0],
            score_mismatch_penalty[1]
        ))
    args = [
        '-x', index_basename,
        '-U', unpaired_reads,
        '-S', out_path,
        '--seed', rng_seed,
        '-D', effort_seed_ext_failures,
        '-R', effort_num_reseeds,
        '-N', aln_seed_num_mismatches,
        '-L', aln_seed_len,
        '-i', aln_seed_interval_fn,
        '--gbar', aln_gbar,
        '--dpad', aln_dpad,
        '--n-ceil', aln_n_ceil,
        '--score-min', score_min_fn,
        '--mp', f"{score_mismatch_penalty[0]},{score_mismatch_penalty[1]}",
        '--rdg', f"{score_read_gap_penalty[0]},{score_read_gap_penalty[1]}",
        '--rfg', f"{score_ref_gap_penalty[0]},{score_ref_gap_penalty[1]}"
    ]

    if score_match_bonus is not None:
        args += ['--ma', score_match_bonus]

    if local:
        args.append('--local')
    else:
        args.append('--end-to-end')

    if quality_format == 'phred33':
        args.append('--phred33')
    elif quality_format == 'phred64':
        args.append('--phred64')
    elif quality_format == 'solexa':
        args.append('--solexa-quals')
    elif quality_format == 'int':
        args.append('--int-quals')
    else:
        raise RuntimeError("Unrecognized quality_format argument `{}`".format(quality_format))

    if report_all_alignments and report_k_alignments is not None:
        raise ValueError("Can't simultaneously use `report_all_alignments` and `report_k_alignments` settings.")

    if report_all_alignments:
        args.append('-a')
    elif report_k_alignments is not None:
        args += ['-k', report_k_alignments]

    if offrate:
        args += ['--offrate', offrate]

    if sam_suppress_noalign:
        args.append('--no-unal')

    if num_threads > 1:
        args += ['--threads', num_threads]

    env = os.environ.copy()
    env['BOWTIE2_INDEXES'] = str(index_basepath)
    logger.debug("Bowtie2 index dir: {}".format(index_basepath))

    exit_code = call_command(
        command_path,
        args=args,
        environment=env
    )

    if exit_code != 0:
        raise CommandLineException("bowtie2", exit_code)
