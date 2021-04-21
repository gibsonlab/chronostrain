import argparse
import csv
import re
import os
from typing import List, Tuple

from chronostrain import logger, cfg
from multiprocessing import cpu_count
from chronostrain.util.sam_handler import SamFlags, SamHandler
from chronostrain.util.external import bwa
from chronostrain.util.external import CommandLineException
from chronostrain.util.external.commandline import call_command


def ref_base_name(ref_path: str) -> str:
    """
    Convert a reference fasta path to the "base name" (typically the accession + marker name).
    e.g. "/data/CP007799.1-16S.fq" -> "CP007799.1-16S"
    """
    ref_fasta_filename = os.path.split(ref_path)[-1]
    ref_base_name = os.path.splitext(ref_fasta_filename)[0]
    return ref_base_name


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


def reconstruct_md_tags(cora_output_paths, reference_paths):
    """
    Uses samtool's mdfill to reconstruct the MD (mismatch and deletion) tag and overwrites the SAM file with the output.
    The original file is preserved, the tag is inserted in each line
    """
    for cora_output_path, reference_path in zip(cora_output_paths, reference_paths):
        exit_code = call_command('samtools', ['fillmd', '-S', cora_output_path, reference_path], output_path=cora_output_path)
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


def filter_file(sam_file, result_metadata_path, result_fq_path, result_sam_path) -> str:
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
    def __init__(self, reference_file_path: str, reads_paths: list, time_points: list, align_cmd: str, output_dir: str):
        logger.debug("Ref path: {}".format(reference_file_path))

        # Note: Bowtie2 does not have the restriction to uncompress bz2 files, but bwa does.
        if reference_file_path.endswith(".bz2"):
            call_command("bz2", args=["-dk", reference_file_path])
            self.reference_file_path = reference_file_path[:-4]
        else:
            self.reference_path = reference_file_path

        self.reads_paths = reads_paths
        self.time_points = time_points
        self.align_cmd = align_cmd
        self.output_dir = output_dir

    def apply_filter(self) -> List[str]:
        """
        :return: A list of paths to the resulting filtered read files.
        """
        if self.align_cmd == 'bwa':
            return self.apply_bwa_filter()
        else:
            raise NotImplementedError("Alignment command `{}` not currently supported.".format(self.align_cmd))

    def apply_bwa_filter(self):
        resulting_files = []

        bwa.bwa_index(reference_path=self.reference_path)

        for time_point, reads_path in zip(self.time_points, self.reads_paths):
            base_path = os.path.dirname(reads_path)
            aligner_tmp_dir = os.path.join(base_path, "tmp")

            if not os.path.exists(aligner_tmp_dir):
                os.makedirs(aligner_tmp_dir)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            sam_path = os.path.join(aligner_tmp_dir, "{}-{}.sam".format(time_point, ref_base_name(self.reference_path)))

            bwa.bwa_mem(output_path=sam_path,
                        reference_path=self.reference_path,
                        read_path=reads_path,
                        min_seed_length=100)

            result_metadata_path = os.path.join(self.output_dir, 'metadata_{}.tsv'.format(time_point))
            result_fq_path = os.path.join(self.output_dir, "reads_{}.fq".format(time_point))
            result_sam_path = os.path.join(self.output_dir, 'alignments_{}.sam'.format(time_point))

            ref_filtered_path = filter_file(sam_path, result_metadata_path, result_fq_path, result_sam_path)
            resulting_files.append(ref_filtered_path)

        save_input_csv(self.time_points, self.output_dir, "input_files.csv", resulting_files)

        return resulting_files


def get_input_paths(base_dir) -> Tuple[List[str], List[float]]:
    time_points = []
    read_files = []

    input_specification_path = os.path.join(base_dir, "input_files.csv")
    try:
        with open(input_specification_path, "r") as f:
            input_specs = csv.reader(f, delimiter=',', quotechar='"')
            for item in input_specs:
                time_points.append(float(item[0]))
                read_files.append(os.path.join(base_dir, item[1]))
    except FileNotFoundError:
        raise FileNotFoundError("Missing required file `input_files.csv` in directory {}.".format(base_dir)) from None

    return read_files, time_points


def save_input_csv(time_points, out_dir, out_filename, read_files):
    with open(os.path.join(out_dir, out_filename), "w") as f:
        for t, read_file in zip(time_points, read_files):
            print("\"{}\",\"{}\"".format(t, read_file), file=f)


def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference on time-series reads.")

    # Input specification.
    parser.add_argument('-r', '--reads_dir', required=True, type=str,
                        help='<Required> Directory containing read files. The directory requires a `input_files.csv` '
                             'which contains information about the input reads and corresponding time points.')
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        dest="output_dir",
                        help='<Required> The file path to save learned outputs to.')

    return parser.parse_args()


def main():
    args = parse_args()
    db = cfg.database_cfg.get_database()
    read_paths, time_points = get_input_paths(args.reads_dir)

    # ============ Perform read filtering.
    logger.info("Performing filter on reads.")
    filt = Filter(
        reference_file_path=db.get_multifasta_file(),
        reads_paths=read_paths,
        time_points=time_points,
        align_cmd=cfg.filter_cfg.align_cmd,
        output_dir=args.output_dir
    )
    _ = filt.apply_filter()
    logger.info("Finished filtering.")


if __name__ == "__main__":
    main()
