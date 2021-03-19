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


class CoraException(BaseException):
    def __init__(self, subcommand, exit_code):
        super().__init__("Cora encountered an error. ({})".format(subcommand))
        self.subcommand = subcommand
        self.exit_code = exit_code


def call_cora(read_length, reference_paths, hom_table_dir, read_path, output_paths, cora_path="cora"):
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
            raise CoraException("Cora encountered an error. (faiGenerate)")

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
            raise CoraException("coraIndex", exit_code)

        # ============= Step 3: mapperIndex
        exit_code = call_command(cora_path, ['mapperIndex',
                                             '--Map', 'BWA',
                                             '--Exec', 'bwa',
                                             reference_path])
        if exit_code != 0:
            raise CoraException("mapperIndex", exit_code)

        # ============= Step 4: readFileGen
        exit_code = call_command(cora_path, ['readFileGen', cora_read_file_path,
                                             '-S',
                                             read_path])
        if exit_code != 0:
            raise CoraException("readFilegen", exit_code)

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
            raise CoraException("search", exit_code)


def call_bwa(reference_paths, read_path, output_paths):
    for reference_path, output_path in zip(reference_paths, output_paths):
        try:
            call_command('bwa', ['index', reference_path])

            call_command('bwa', ['mem', '-o', output_path, reference_path, read_path])
        # p = check_output(['bwa', 'mem', reference_path, read_path, '>', output_path])
        except Exception as e:
            logger.error(e)
            exit(1)


def reconstruct_md_tags(cora_output_paths, reference_paths):
    """
    Uses samtool's mdfill to reconstruct the MD (mismatch and deletion) tag and overwrites the SAM file with the output.
    The original file is preserved, the tag is inserted in each line
    """
    for cora_output_path, reference_path in zip(cora_output_paths, reference_paths):
        try:
            call_command('samtools', ['fillmd', '-S', cora_output_path, reference_path, '>', cora_output_path])
        except Exception as e:
            logger.error(e)
            exit(1)


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
                print("Unrecognized sequence in MD tag: " + sequence)
    return total_matches / total_clipped_length


def find_beginning_clip(cigar_tag):
    split_cigar = re.findall('\d+|\D+', cigar_tag)
    if split_cigar[1] == 'S':
        return int(split_cigar[0])
    return 0


def apply_filter(percent_identity, beginning_clip, start_index):
    """
    Applies a filtering criteria for reads that continue in the pipeline. Currently a simple threshold on percent identity,
    likely should be adjusted to maximize downstream sensitivity?
    Also filters out alignments that begin mid-read
    """
    return int(percent_identity > 0.9 and (beginning_clip > start_index or beginning_clip < 10))


def filter_file(sam_file, output_base_path):
    """
    Parses a sam file and filters reads using the above criteria. Writes the results to a fastq file containing the passing
    reads and a TSV containing columns:
    Read Name    Percent Identity    Passes Filter
    """
    try:
        sam_file = open(sam_file, 'r')
        result_metadata = open(os.path.join(output_base_path, 'Metadata.tsv'), 'w')
        result_fq = open(os.path.join(output_base_path, 'Reads.fq'), 'w')
        result_full_alignment = open(os.path.join(output_base_path, 'Alignments.sam'), 'w')
    except IOError as e:
        logger.error(e)
        exit(1)

    for aln in sam_file:
        # Header line. Skip.
        if aln[0] == '@':
            continue

        tags = aln.strip().split('\t')
        start_index = int(tags[3])
        for tag in tags:
            print(tag)
            if tag[:5] == 'MD:Z:':
                percent_identity = parse_md_tag(tag[5:])
                result_metadata.write(tags[0] + '\t{:0.4f}\t'.format(percent_identity) + str(
                    apply_filter(percent_identity, find_beginning_clip(tags[5]), start_index)) + '\n')
                if apply_filter(percent_identity, find_beginning_clip(tags[5]), start_index) == 1:
                    result_fq.write('@' + tags[0] + '\n')  # Read info
                    result_fq.write(tags[9] + '\n')  # Read sequence
                    result_fq.write('+\n')
                    result_fq.write(tags[10] + '\n')  # Read quality
                    result_full_alignment.write(aln)

    result_full_alignment.close()
    result_metadata.close()
    result_fq.close()


class Filter:
    def __init__(self, reference_file_paths: list, reads_paths: list, time_points: list, cora_path: str):
        self.reference_paths = [os.path.join(os.getcwd(), path) for path in reference_file_paths]
        self.reads_paths = reads_paths
        self.time_points = time_points
        self.cora_path = cora_path

    @staticmethod
    def cleanup_intermediate_files(filenames: list):
        for file in filenames:
            try:
                _ = subprocess.check_output(['rm', '-f', file])
            except Exception as e:
                logger.error(e)
                exit(1)

    def cat_resulting_reads(self, output_path: str, filenames: list, time_point):
        # passed_read_fastq_path = os.path.join(output_basepath, "{}-passed-reads.fq".format(time_point))
        with open(output_path, 'w') as output:
            for file in filenames:
                with open(file, 'r') as input:
                    for line in input:
                        output.write(line)
        return output_path

    def apply_filter(self, read_length: int):
        resulting_files = []
        for time_point, reads_path in zip(self.time_points, self.reads_paths):
            base_path = os.path.dirname(reads_path)
            cora_tmp_dir = os.path.join(base_path, "tmp", "cora")
            filtered_reads_base_path = os.path.join(base_path, "filtered")

            if not os.path.exists(cora_tmp_dir):
                os.makedirs(cora_tmp_dir)
            if not os.path.exists(filtered_reads_base_path):
                os.makedirs(filtered_reads_base_path)

            sam_paths = [
                os.path.join(
                    cora_tmp_dir,
                    "{}.sam".format(ref_base_name(reference_path))
                )
                for reference_path in self.reference_paths
            ]

            call_cora(
                read_length=read_length,
                reference_paths=self.reference_paths,
                hom_table_dir=cora_tmp_dir,
                read_path=reads_path,
                output_paths=sam_paths,
                cora_path=self.cora_path
            )

            for reference_path, sam_path in zip(self.reference_paths, sam_paths):
                ref_base = ref_base_name(reference_path)
                output_dir = os.path.join(filtered_reads_base_path, "{}-{}-Passed".format(time_point, ref_base))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filter_file(sam_path, output_dir)

            print("DONE")
            exit(1)

            # resulting_files.append(self.cat_resulting_reads(
            #     output_path="{}-filtered.fq".format(),
            #     [
            #         os.path.join(base_path, "{}-{}-filtered.fq".format(
            #             self.ref_base_name(reference_path),
            #             time_point
            #         ))
            #         for reference_path in self.reference_paths
            #     ],
            #     time_point
            # ))
        return resulting_files
