from pathlib import Path
from typing import List, Optional

from chronostrain.logging import create_logger
from .commandline import call_command, CommandLineException

logger = create_logger(__name__)


def dashing2_sketch(
        input_fasta: Optional[List[Path]] = None,
        filename_list_input: Optional[Path] = None,
        comparison_out: Optional[Path] = None,
        cache_sketches: bool = True,
        sketch_prefix: Optional[Path] = None,
        sketch_outfile: Optional[Path] = None,
        min_hash_mode: str = 'BagMinHash',
        kmer_length: int = 32,  # k-mer length.
        seed: Optional[int] = None,
        window_size: Optional[int] = None,
        kmer_spacing: Optional[str] = None,
        use_long_kmers: bool = False,
        square_distance_matrix: bool = False,
        emit_distances: bool = False,
        binary_output: bool = False,
        sketch_size: int = 1024,
        silent: bool = False
):
    """
    :param input_fasta: A list of input FASTA paths to sketch.
    :param filename_list_input: A file containing filenames (one per line) of FASTA files to sketch.
    :param comparison_out: If specified, will perform an all-to-all distance computation and output to this file.
    :param cache_sketches: If True, then sketches are stored to file. By default, sketches are stored next to the original fasta.
    :param sketch_prefix: If specified (and cache is enabled), store sketches using this path prefix instead of the default.
    :param min_hash_mode: 'FullSet', 'ProbMinHash' or 'BagMinHash'
    :param kmer_length: The length of the k-mers to use.
    :param seed: the random seed to use for the hash functions/minimizer selection.
    :param window_size: larger windows yield fewer minimizers. default is kmer length.
    :param kmer_spacing: default is unspaced k-mer minimizer patterns. Allows for some positions to be ignored.
    :param use_long_kmers: Use 128-bit k-mer hashes instead of 64-bit
    :param sketch_size: The size of the sketch to use (number of minimizers).
    :return:
    """
    if input_fasta is None and filename_list_input is None:
        raise ValueError(
            "dashing2_sketch requires at least one input argument (either `input_fasta` or `filename_list_input`)."
        )
    if input_fasta is not None and filename_list_input is not None:
        raise ValueError(
            "dashing2_sketch does not support simultaneous input of `input_fasta` and `filename_list_input`."
        )

    args = [
        'sketch',
        '-k', kmer_length,
        '--sketchsize', sketch_size
    ]

    # options
    if seed is not None:
        args += ['--seed', seed]
    if window_size is not None:
        args += ['-w', window_size]
    if kmer_spacing is not None:
        args += ['--spacing', kmer_spacing]
    if use_long_kmers:
        args += ['--long-kmers']
    if comparison_out is not None:
        args += ['--cmpout', comparison_out]
    if cache_sketches:
        args += ['--cache']
    if sketch_prefix is not None:
        args += ['--outprefix', sketch_prefix]
    if sketch_outfile is not None:
        args += ['-o', sketch_outfile]

    if min_hash_mode == 'FullSet':
        args += ['--full']
    elif min_hash_mode == 'BagMinHash':
        args += ['--bagminhash']
    elif min_hash_mode == 'ProbMinHash':
        args += ['--pminhash']
    elif min_hash_mode == 'FullCountDict':
        args += ['--countdict']
    else:
        raise ValueError(f"Unrecognized argument value `{min_hash_mode}` for min_hash_mode")

    if emit_distances:
        args += ['--distance']
    if binary_output:
        args += ['--binary-output']
    if square_distance_matrix:
        args += ['--square']

    # input files
    if input_fasta is not None:
        args += input_fasta
    elif filename_list_input is not None:
        args += ['-F', filename_list_input]
    else:
        raise RuntimeError("Unexpected error; else scenario should have been already handled.")

    exit_code = call_command(
        'dashing2',
        args=args,
        silent=silent
    )
    if exit_code != 0:
        raise CommandLineException("dashing2 sketch", exit_code)