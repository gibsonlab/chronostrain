from .commandline import *


def bwa_index(reference_path, bwa_path="bwa"):
    '''
    TODO: Allow configurable minimum seed length
    '''

    exit_code = call_command(bwa_path, ['index', reference_path])
    if exit_code != 0:
        raise CommandLineException("bwa index", exit_code)


def bwa_mem(output_path: str,
            reference_path: str,
            read_path: str,
            min_seed_length: int,
            bwa_path="bwa"):
    exit_code = call_command(
        command=bwa_path,
        args=[
            'mem',
            '-o', output_path,
            '-k', str(min_seed_length),
            reference_path,
            read_path
        ]
    )
    if exit_code != 0:
        raise CommandLineException("bwa mem", exit_code)
