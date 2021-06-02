import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

from chronostrain import logger, cfg
from multiprocessing import cpu_count
from chronostrain.util.external import bwa
from chronostrain.util.external import CommandLineException
from chronostrain.util.external.commandline import call_command
from chronostrain.util.sam_handler import SamHandler


def ref_base_name(ref_path: Path) -> str:
    """
    Convert a reference fasta path to the "base name" (typically the accession + marker name).
    e.g. "/data/CP007799.1-16S.fq" -> "CP007799.1-16S"
    """
    return ref_path.with_suffix('').name


def call_cora(read_length: int,
              reference_paths: List[Path],
              hom_table_dir: Path,
              read_path: Path,
              output_paths: List[Path],
              cora_path="cora"):
    """
    Slightly obfuscated, not updated for multifasta input
    """
    threads_available = cpu_count()
    read_path_dir = read_path.parent
    cora_read_file_path = read_path_dir / "coraReadFileList"

    for reference_path, output_path in zip(reference_paths, output_paths):
        ref_base = ref_base_name(reference_path)
        hom_exact_path = hom_table_dir / "{}.exact".format(ref_base)
        hom_inexact_path = hom_table_dir / "{}.inexact".format(ref_base)

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


def reconstruct_md_tags(cora_output_paths: List[Path], reference_paths: List[Path]):
    """
    Uses samtool's mdfill to reconstruct the MD (mismatch and deletion) tag and overwrites the SAM file with the output.
    The original file is preserved, the tag is inserted in each line
    """
    for cora_output_path, reference_path in zip(cora_output_paths, reference_paths):
        exit_code = call_command('samtools', ['fillmd', '-S', cora_output_path, reference_path],
                                 output_path=cora_output_path)
        if exit_code != 0:
            raise CommandLineException("samtools fillmd", exit_code)


def parse_md_tag(tag: str):
    """
    Calculate the percent identity from a clipped MD tag. Three types of subsequences are read:
    (1) Numbers represent the corresponding amount of sequential matches
    (2) Letters represent a mismatch and two sequential mismatches are separated by a 0
    (3) A ^ represents a deletion and will be followed by a sequence of consecutive letters
        corresponding to the bases missing
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
    """
    TODO: This trim is for HiSeq150, avg phred score of 30. Add setup in config, possibly by reading quality profile
    """
    return read_quality[5:-10]


def probability_from_ascii_encoding(ascii_quality):
    return 10**(-(ord(ascii_quality)-33)/10)


def find_expected_errors(probs):
    return sum(probs)


def filter_on_read_quality(read_quality, error_threshold=3):
    """
    TODO: Allow configurable expected error threshold
    """
    # trimmed_quality = trim_read_quality(read_quality)
    probs = [probability_from_ascii_encoding(ascii_quality) for ascii_quality in read_quality]
    expected_errs = find_expected_errors(probs)
    if expected_errs > error_threshold:
        return False
    return True


def filter_on_match_identity(percent_identity, identity_threshold=0.9):
    """
    Applies a filtering criteria for reads that continue in the pipeline.
    Currently a simple threshold on percent identity, likely should be adjusted to maximize downstream sensitivity?
    """
    return percent_identity > identity_threshold


def filter_file(sam_file: Path,
                reference_path: Path,
                result_metadata_path: Path,
                result_fq_path: Path,
                result_sam_path: Path):
    """
    Parses a sam file and filters reads using the above criteria.
    Writes the results to a fastq file containing the passing reads and a TSV containing columns:
        Read Name    Percent Identity    Passes Filter?
    """

    result_metadata = open(result_metadata_path, 'w')
    result_fq = open(result_fq_path, 'w')
    result_full_alignment = open(result_sam_path, 'w')

    sam_handler = SamHandler(sam_file, reference_path)
    for sam_line in sam_handler.mapped_lines():
        if sam_line.optional_tags['MD'] is not None:
            percent_identity = parse_md_tag(sam_line.optional_tags['MD'])
            passed_filter = (
                filter_on_match_identity(percent_identity) and filter_on_read_quality(sam_line.quality)
            )
            result_metadata.write(
                sam_line.readname
                + '\t{:0.4f}\t'.format(percent_identity)
                + str(int(passed_filter))
                + '\n'
            )
            if passed_filter:
                result_fq.write('@' + sam_line.readname + '\n')
                result_fq.write(sam_line.read + '\n')
                result_fq.write('+\n')
                result_fq.write(sam_line.quality + '\n')
                result_full_alignment.write(str(sam_line))

    result_full_alignment.close()
    result_metadata.close()
    result_fq.close()

class Filter:
    def __init__(self,
                 reference_file_path: Path,
                 reads_paths: List[Path],
                 time_points: List[float],
                 align_cmd: str,
                 output_dir: Path):
        logger.debug("Ref path: {}".format(reference_file_path))

        # Note: Bowtie2 does not have the restriction to uncompress bz2 files, but bwa does.
        if reference_file_path.suffix == ".bz2":
            call_command("bz2", args=["-dk", reference_file_path])
            self.reference_path = reference_file_path.with_suffix('')
        else:
            self.reference_path = reference_file_path

        self.reads_paths = reads_paths
        self.time_points = time_points
        self.align_cmd = align_cmd
        self.output_dir = output_dir

    def apply_filter(self):
        """
        :return: A list of paths to the resulting filtered read files.
        """
        if self.align_cmd == 'bwa':
            self.apply_bwa_filter()
        else:
            raise NotImplementedError("Alignment command `{}` not currently supported.".format(self.align_cmd))

    def apply_bwa_filter(self):
        bwa.bwa_index(reference_path=self.reference_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        resulting_files = []
        for time_point, reads_path in zip(self.time_points, self.reads_paths):
            base_path = reads_path.parent
            aligner_tmp_dir = base_path / "tmp"
            aligner_tmp_dir.mkdir(parents=True, exist_ok=True)

            sam_path = aligner_tmp_dir / "{}-{}.sam".format(time_point, ref_base_name(self.reference_path))

            bwa.bwa_mem(output_path=sam_path,
                        reference_path=self.reference_path,
                        read_path=reads_path,
                        min_seed_length=100)

            result_metadata_path = self.output_dir / 'metadata_{}.tsv'.format(time_point)
            result_fq_path = self.output_dir / "reads_{}.fq".format(time_point)
            result_sam_path = self.output_dir / 'alignments_{}.sam'.format(time_point)

            filter_file(sam_path, self.reference_path, result_metadata_path, result_fq_path, result_sam_path)
            logger.info("Timepoint {t}, filtered reads file: {f}".format(
                t=time_point, f=result_fq_path
            ))
            resulting_files.append(result_fq_path)

        save_input_csv(self.time_points, self.output_dir, "input_files.csv", resulting_files)


def get_input_paths(base_dir: Path) -> Tuple[List[Path], List[float]]:
    time_points = []
    read_files = []

    input_specification_path = base_dir / "input_files.csv"
    try:
        with open(input_specification_path, "r") as f:
            input_specs = csv.reader(f, delimiter=',', quotechar='"')
            for item in input_specs:
                time_points.append(float(item[0]))
                read_files.append(base_dir / item[1])
    except FileNotFoundError:
        raise FileNotFoundError("Missing required file `input_files.csv` in directory {}.".format(base_dir)) from None

    return read_files, time_points


def save_input_csv(time_points, out_dir: Path, out_filename, read_files):
    with open(out_dir / out_filename, "w") as f:
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
    read_paths, time_points = get_input_paths(Path(args.reads_dir))

    # ============ Perform read filtering.
    logger.info("Performing filter on reads.")
    filt = Filter(
        reference_file_path=db.multifasta_file,
        reads_paths=read_paths,
        time_points=time_points,
        align_cmd=cfg.filter_cfg.align_cmd,
        output_dir=Path(args.output_dir)
    )
    filt.apply_filter()
    logger.info("Finished filtering.")


if __name__ == "__main__":
    main()
