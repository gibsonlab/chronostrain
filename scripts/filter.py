import re
import os
import subprocess
from typing import List

from chronostrain import logger
from multiprocessing import cpu_count


def ref_base_name(ref_path: str) -> str:
    """
    Convert a reference fasta path to the "base name" (typically the accession + marker name).
    e.g. "/data/CP007799.1-16S.fq" -> "CP007799.1-16S"
    """
    ref_fasta_filename = os.path.split(ref_path)[-1]
    ref_base_name = os.path.splitext(ref_fasta_filename)[0]
    return ref_base_name


def call_command(command: str, args: List[str]) -> int:
    """
    Executes the command (using the subprocess module).
    :param command: The binary to run.
    :param args: The command-line arguments.
    :return: The exit code. (zero by default, the program's returncode if error.)
    """
    logger.debug("EXECUTE: {} {}".format(
        command,
        " ".join(args)
    ))

    p = subprocess.run([command] + args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    logger.debug("STDOUT: {}".format(p.stdout.decode("utf-8")))
    logger.debug("STDERR: {}".format(p.stderr.decode("utf-8")))
    return p.returncode


def call_command_to_file(command: str, args: List[str], output_path: str) -> int:
    '''
    Currently used only by samtools, which does not have optional output redirection
    '''
    logger.debug("EXECUTE: {} {}".format(
        command,
        " ".join(args)
    ))

    # Done in two steps to replicate UNIX pipes, which overwrite after read
    p = subprocess.run([command] + args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output_file = open(output_path, 'w')
    output_file.write(p.stdout.decode("utf-8"))
    output_file.close()

    logger.debug("STDERR: {}".format(p.stderr.decode("utf-8")))
    return p.returncode


class CommandLineException(BaseException):
    def __init__(self, cmd, exit_code):
        super().__init__("`{}` encountered an error.".format(cmd))
        self.cmd = cmd
        self.exit_code = exit_code


def call_cora(read_length, reference_paths, hom_table_dir, read_path, output_paths, cora_path="cora"):
    '''
    Slightly obfuscated, not updated for multifasta input
    '''
    threads_available = cpu_count()
    read_path_dir = os.path.dirname(read_path)
    cora_read_file_path = os.path.join(read_path_dir, "coraReadFileList")

    for reference_path, output_path in zip(reference_paths, output_paths):
        ref_base = ref_base_name(reference_path)
        hom_exact_path = os.path.join(hom_table_dir, "{}.exact".format(ref_base))
        hom_inexact_path = os.path.join(hom_table_dir, "{}.inexact".format(ref_base))

        # ============= Step 1: faiGenerate
        exit_code = call_command(cora_path, ['faiGenerate', reference_path])
        if exit_code != 0:
            raise CommandLineException("cora faiGenerate", exit_code)

        # ============= Step 2: coraIndex
        exit_code = call_command(cora_path, ['coraIndex',
                                             '-K', '50',
                                             '-p', '10',
                                             '-t', str(threads_available),
                                             reference_path,
                                             hom_exact_path,
                                             hom_inexact_path])
        if exit_code == 8:
            logger.debug("HomTable files already exist (Exit code 8). Skipping this step.")
        elif exit_code != 0:
            raise CommandLineException("cora coraIndex", exit_code)

        # ============= Step 3: mapperIndex
        exit_code = call_command(cora_path, ['mapperIndex',
                                             '--Map', 'BWA',
                                             '--Exec', 'bwa',
                                             reference_path])
        if exit_code != 0:
            raise CommandLineException("cora mapperIndex", exit_code)

        # ============= Step 4: readFileGen
        exit_code = call_command(cora_path, ['readFileGen', cora_read_file_path,
                                             '-S',
                                             read_path])
        if exit_code != 0:
            raise CommandLineException("cora readFilegen", exit_code)

        # ============= Step 5: search
        exit_code = call_command(cora_path, ['search',
                                             '-C', '1111',
                                             '--Mode', 'BEST',
                                             '--Map', 'BWA',
                                             '--Exec', 'bwa',
                                             '-R', 'SINGLE',
                                             '-O', output_path,
                                             '-L', str(read_length),
                                             cora_read_file_path,
                                             reference_path,
                                             hom_exact_path,
                                             hom_inexact_path])
        if exit_code != 0:
            raise CommandLineException("cora search", exit_code)

    reconstruct_md_tags(output_paths, reference_paths)


def call_bwa(reference_path, read_path, output_path, bwa_path="bwa"):
    '''
    TODO: Allow configurable minimum seed length
    '''
    MIN_SEED_LENGTH = 100

    exit_code = call_command(bwa_path, ['index', reference_path])
    if exit_code != 0:
        raise CommandLineException("bwa index", exit_code)

    exit_code = call_command(bwa_path, ['mem', '-o', output_path, '-k', str(MIN_SEED_LENGTH), reference_path, read_path])
    if exit_code != 0:
        raise CommandLineException("bwa mem", exit_code)


def reconstruct_md_tags(cora_output_paths, reference_paths):
    """
    Uses samtool's mdfill to reconstruct the MD (mismatch and deletion) tag and overwrites the SAM file with the output.
    The original file is preserved, the tag is inserted in each line
    """
    for cora_output_path, reference_path in zip(cora_output_paths, reference_paths):
        exit_code = call_command_to_file('samtools', ['fillmd', '-S', cora_output_path, reference_path], cora_output_path)
        if exit_code != 0:
            raise CommandLineException("samtools fillmd", exit_code)


def parse_md_tag(tag):
    """
    Calculate the percent identity from a clipped MD tag. Three types of subsequences are read:
    (1) Numbers represent the corresponding amount of sequential matches
    (2) Letters represent a mismatch and two sequential mismatches are separated by a 0
    (3) A ^ represents a deletion and will be followed by a sequence of consecutive letters corresponding to the bases missing
    Dividing (1) by (1)+(2)+(3) will give matches/clipped_length, or percent identity
    """
    split_md = re.findall('\d+|\D+', tag)
    total_clipped_length = 0
    total_matches = 0
    for sequence in split_md:
        if sequence.isnumeric():  # (1)
            total_clipped_length += int(sequence)
            total_matches += int(sequence)
        else:
            if sequence[0] == '^':  # (3)
                total_clipped_length += len(sequence) - 1
            elif len(sequence) == 1:  # (2)
                total_clipped_length += 1
            else:
                logger.warn("Unrecognized sequence in MD tag: " + sequence)
    return total_matches / total_clipped_length


def trim_read_quality(read_quality):
    '''
    TODO: This trim is for HiSeq150, avg phred score of 30. Add setup in config, possibly by reading quality profile
    '''
    return read_quality[5:-10]


def probability_from_ascii_encoding(ascii_quality):
    '''
    TODO: Double check the encoding for other sequencers. This agrees with fastQC on illumina reads
    '''
    return 10**(-(ord(ascii_quality)-33)/10)


def find_expected_errors(probs):
    return sum(probs)


def filter_on_read_quality(read_quality):
    '''
    TODO: Allow configurable expected error threshold
    '''
    ERROR_THRESHOLD = 3

    trimmed_quality = trim_read_quality(read_quality)
    probs = [probability_from_ascii_encoding(ascii_quality) for ascii_quality in read_quality]
    expected_errs = find_expected_errors(probs)
    if expected_errs > ERROR_THRESHOLD:
        return False
    return True


def filter_on_match_identity(percent_identity):
    """
    Applies a filtering criteria for reads that continue in the pipeline. Currently a simple threshold on percent identity,
    likely should be adjusted to maximize downstream sensitivity?

    TODO: Allow configurable percent identity threshold
    """
    return percent_identity > 0.9


def filter_file(sam_file, output_path_stem) -> str:
    """
    Parses a sam file and filters reads using the above criteria.
    Writes the results to a fastq file containing the passing reads and a TSV containing columns:
        Read Name    Percent Identity    Passes Filter?
    :return: The full path to the relevant reads fastq path.
    """

    # SAM file tag indices. Optional tags like MD can appear in any order after index 10
    READ_ACCESSION_INDEX = 0
    MAPPING_FLAG_INDEX = 1
    READ_INDEX = 9
    QUALITY_INDEX = 10

    result_metadata_path = output_path_stem + 'metadata.tsv'
    result_fq_path = output_path_stem + 'reads.fq'
    result_sam_path = output_path_stem + 'Alignments.sam'

    sam_file = open(sam_file, 'r')
    result_metadata = open(result_metadata_path, 'w')
    result_fq = open(result_fq_path, 'w')
    result_full_alignment = open(result_sam_path, 'w')

    for aln in sam_file:
        # Header line. Skip.
        if aln[0] == '@':
            continue
    
        tags = aln.strip().split('\t')

        # Unmapped flag, no alignment.
        if tags[MAPPING_FLAG_INDEX] == '4':
            continue

        for tag in tags:
            if tag[:5] == 'MD:Z:':
                percent_identity = parse_md_tag(tag[5:])
                passed_filter = bool(filter_on_match_identity(percent_identity) \
                    and filter_on_read_quality(tags[QUALITY_INDEX]))

                result_metadata.write(tags[READ_ACCESSION_INDEX] + '\t{:0.4f}\t'.format(percent_identity)
                    + str(int(passed_filter)) + '\n')
                if passed_filter:
                    result_fq.write('@' + tags[READ_ACCESSION_INDEX] + '\n')
                    result_fq.write(tags[READ_INDEX] + '\n')
                    result_fq.write('+\n')
                    result_fq.write(tags[QUALITY_INDEX] + '\n')
                    result_full_alignment.write(aln)

    result_full_alignment.close()
    result_metadata.close()
    result_fq.close()
    sam_file.close()
    return result_fq_path


class Filter:
    def __init__(self, reference_file_path: str, reads_paths: list, time_points: list, align_cmd: str):
        logger.debug("Ref path: {}".format(reference_file_path))
        self.reference_path = os.path.join(os.getcwd(), reference_file_path)
        self.reads_paths = reads_paths
        self.time_points = time_points
        self.align_cmd = align_cmd

    def apply_filter(self, read_length: int):
        resulting_files = []
        for time_point, reads_path in zip(self.time_points, self.reads_paths):
            base_path = os.path.dirname(reads_path)
            aligner_tmp_dir = os.path.join(base_path, "tmp")
            filtered_reads_dir = os.path.join(base_path, "filtered")
            final_filtered_reads_path = os.path.join(base_path, "{}.filtered.fq".format(time_point))

            if not os.path.exists(aligner_tmp_dir):
                os.makedirs(aligner_tmp_dir)
            if not os.path.exists(filtered_reads_dir):
                os.makedirs(filtered_reads_dir)

            sam_path = os.path.join(aligner_tmp_dir, "{}-{}.sam".format(time_point, ref_base_name(self.reference_path)))

            call_bwa(
                reference_path=self.reference_path,
                read_path=reads_path,
                output_path=sam_path,
                bwa_path=self.align_cmd
            )

            ref_filtered_path = filter_file(sam_path, filtered_reads_dir + "/{}-".format(time_point))
            resulting_files.append(ref_filtered_path)

        return resulting_files
