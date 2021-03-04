import sys
import re
import os
from chronostrain import logger
from multiprocessing import cpu_count
from subprocess import check_output


def call_cora(read_length, reference_paths, hom_table_path, read_path, output_paths):
	threads_available = cpu_count()
	
	for reference_path, output_path in zip(reference_paths, output_paths):
		try:
			p = check_output(['cora', 'coraIndex', '-K', '50', '-p', '10', '-t', str(threads_available), reference_path, \
				hom_table_path + 'exact', hom_table_path + 'inexact'])
			
			p = check_output(['cora', 'mapperIndex', '--Map', 'BWA', '--Exec', 'bwa', reference_path])
		
			p = check_output(['cora', 'readFileGen', read_path + 'coraReadFileList', '-P', read_path + '1.fq', read_path + '2.fq'])
			
			p = check_output(['cora', 'search', '-C', '1111', '--Mode', 'BEST', '--Map', 'BWA', '--Exec', 'bwa', '-R', 'PAIRED', \
				'-O', output_path, '-L', str(read_length), read_path + 'coraReadFileList', reference_path, hom_table_path + \
				'exact', hom_table_path + 'inexact'])
			
		except Exception as e:
			logger.error(e)
			sys.exit()


def call_bwa(reference_paths, read_path, output_paths):
	for reference_path, output_path in zip(reference_paths, output_paths):
		try:
			p = check_output(['bwa', 'index', reference_path])

			p = check_output(['bwa', 'mem', '-o', output_path, reference_path, read_path])
			#p = check_output(['bwa', 'mem', reference_path, read_path, '>', output_path])
		except Exception as e:
			logger.error(e)
			sys.exit()


def reconstruct_md_tags(cora_output_paths, reference_paths):
	'''
	Uses samtool's mdfill to reconstruct the MD (mismatch and deletion) tag and overwrites the SAM file with the output.
	The original file is preserved, the tag is inserted in each line
	'''
	for cora_output_path, reference_path in zip(cora_output_paths, reference_paths):
		try:
			p = check_output(['samtools', 'fillmd', '-S', cora_output_path, reference_path, '>', cora_output_path])
		except Exception as e:
			logger.error(e)
			sys.exit()


def parse_md_tag(tag):
	'''
	Calculate the percent identity from a clipped MD tag. Three types of subsequences are read:
	(1) Numbers represent the corresponding amount of sequential matches
	(2) Letters represent a mismatch and two sequential mismatches are separated by a 0
	(3) A ^ represents a deletion and will be followed by a sequence of consecutive letters corresponding to the bases missing
	Dividing (1) by (1)+(2)+(3) will give matches/clipped_length, or percent identity
	'''
	split_md = re.findall('\d+|\D+',tag)
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
	return total_matches/total_clipped_length


def find_beginning_clip(cigar_tag):
	split_cigar = re.findall('\d+|\D+',cigar_tag)
	if split_cigar[1] == 'S':
		return int(split_cigar[0])
	return 0


def apply_filter(percent_identity, beginning_clip, start_index):
	'''
	Applies a filtering criteria for reads that continue in the pipeline. Currently a simple threshold on percent identity,
	likely should be adjusted to maximize downstream sensitivity?
	Also filters out alignments that begin mid-read
	'''
	return int(percent_identity > 0.9 and (beginning_clip > start_index or beginning_clip < 10))


def filter_file(sam_file, output_base_path):
	'''
	Parses a sam file and filters reads using the above criteria. Writes the results to a fastq file containing the passing
	reads and a TSV containing columns:
	Read Name    Percent Identity    Passes Filter
	'''
	try:
		sam_file = open(sam_file, 'r')
		result_metadata = open(output_base_path + 'Metadata.tsv', 'w')
		result_fq = open(output_base_path + 'Reads.fq', 'w')
		result_full_alignment = open(output_base_path + 'Alignments.sam', 'w')
	except IOError as e:
		logger.error(e)
		sys.exit()

	aln = sam_file.readline()
	while aln:
		if aln[0] == '@': # Header lines
			aln = sam_file.readline()
			continue
		tags = aln.strip().split('\t')
		start_index = int(tags[3])
		for tag in tags:
			if tag[:5] == 'MD:Z:':
				percent_identity = parse_md_tag(tag[5:])
				result_metadata.write(tags[0] + '\t{:0.4f}\t'.format(percent_identity) + str(apply_filter(percent_identity, find_beginning_clip(tags[5]), start_index)) + '\n')
				if apply_filter(percent_identity, find_beginning_clip(tags[5]), start_index) == 1:
					result_fq.write('@' + tags[0] + '\n') #Read info
					result_fq.write(tags[9] + '\n') #Read sequence
					result_fq.write('+\n')
					result_fq.write(tags[10] + '\n') #Read quality
					result_full_alignment.write(aln)
		aln = sam_file.readline()

	result_full_alignment.close()
	result_metadata.close()
	result_fq.close()


class Filter:
	def __init__(self, reference_file_paths: list, read_base_path: str, reads_filenames: list, time_points: list):
		self.base_path = os.getcwd() + '/' + read_base_path
		self.reference_paths = [os.getcwd() + '/' + path for path in reference_file_paths]
		self.reads_filenames = reads_filenames
		self.time_points = time_points

	def cleanup_intermediate_files(self, filenames: list):
		for file in filenames:
			try:
				p = check_output(['rm', '-f', file])
			except Exception as e:
				logger.error(e)
				sys.exit()

	def cat_resulting_reads(self, filenames: list, time_point):
		with open(self.base_path + str(time_point) + '-passed_reads.fq', 'w') as output:
			for file in filenames:
				with open(file, 'r') as input:
					for line in input:
						output.write(line)
		return self.base_path + str(time_point) + '-passed_reads.fq'

	def apply_filter(self, read_length: int):
		resulting_files = []
		for time_point, reads_filename in zip(self.time_points, self.reads_filenames):
			intermediate_files = [self.base_path + 'tmp_' + reference_path.split('/')[-1][:-6] + '.sam' for reference_path in self.reference_paths]
			for f in intermediate_files:
				file = open(f, 'w')
				file.close()
			call_cora(read_length, self.reference_paths, self.base_path + 'hom_tables/', self.base_path + reads_filename, intermediate_files)
			for reference_path in self.reference_paths:
				filter_file(self.base_path + 'tmp_' + reference_path.split('/')[-1][:-6] + '.sam', self.base_path + str(time_point) + '-Passed' + reference_path.split('/')[-1][:-6])
			self.cleanup_intermediate_files(intermediate_files)
			resulting_files.append(self.cat_resulting_reads([self.base_path + str(time_point) + '-Passed' + reference_path.split('/')[-1][:-6] + 'Reads.fq' for reference_path in self.reference_paths], time_point))
		return resulting_files
