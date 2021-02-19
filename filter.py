import sys
import re
from util.io.logger import logger
from multiprocessing import cpu_count
from subprocess import check_output

def call_cora(read_length, reference_path, hom_table_path, read_path, output_path):
	threads_available = cpu_count()
	
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

'''
Uses samtool's mdfill to reconstruct the MD (mismatch and deletion) tag and overwrites the SAM file with the output.
The original file is preserved, the tag is inserted in each line
'''	
def reconstruct_md_tags(cora_output_path, reference_path):
	try:
		p = check_output(['samtools', 'fillmd', '-S', cora_output_path, reference_path, '>', cora_output_path])
	except Exception as e:
		logger.error(e)
		sys.exit()

'''
Calculate the percent identity from a clipped MD tag. Three types of subsequences are read:
(1) Numbers represent the corresponding amount of sequential matches
(2) Letters represent a mismatch and two sequential mismatches are separated by a 0
(3) A ^ represents a deletion and will be followed by a sequence of consecutive letters corresponding to the bases missing
Dividing (1) by (1)+(2)+(3) will give matches/clipped_length, or percent identity
'''
def parse_md_tag(tag):
	split_md = re.findall('\d+|\D+',tag)
	total_clipped_length = 0
	total_matches = 0
	for sequence in split_md:
		if sequence.isnumeric(): # (1)
			total_clipped_length += int(sequence)
			total_matches += int(sequence)
		else:
			if sequence[0] == '^': # (3)
				total_clipped_length += len(sequence) - 1
			elif len(sequence) == 1: # (2)
				total_clipped_length += 1
			else:
				print("Unrecognized sequence in MD tag: " + sequence)
	return total_matches/total_clipped_length

'''
Applies a filtering criteria for reads that continue in the pipeline. Currently a simple threshold on percent identity,
likely should be adjusted to maximize downstream sensitivity?
'''
def apply_filter(percent_identity):
	return int(percent_identity > 0.9)

'''
Parses a sam file and filters reads using the above criteria. Writes the results to a fastq file containing the passing
reads and a TSV containing columns:
Read Name    Percent Identity    Passes Filter
'''
def filter_file(sam_file, result_metadata, result_fq):
	aln = sam_file.readline()
	while aln:
		if aln[0] == '@': # Header lines
			aln = sam_file.readline()
			continue
		tags = aln.strip().split('\t')
		for tag in tags:
			if tag[:5] == 'MD:Z:':
				percent_identity = parse_md_tag(tag[5:])
				result_metadata.write(tags[0] + '\t{:0.4f}\t'.format(percent_identity) + str(apply_filter(percent_identity)) + '\n')
				result_fq.write('@' + tags[0] + '\n') #Read info
				result_fq.write(tags[9] + '\n') #Read sequence
				result_fq.write('+\n')
				result_fq.write(tags[10] + '\n') #Read quality
				aln = sam_file.readline()
		aln = sam_file.readline()
	result_metadata.close()
	result_fq.close()

def parse_args():
	if not len(sys.argv) == 8:
		print("Unexpected number of arguments")
		sys.exit()
	try:
		open(sys.argv[2], 'r')
		open(sys.argv[4] + '1.fq', 'r')
		open(sys.argv[4] + '2.fq', 'r')
	except IOError as e:
		print(e)
		sys.exit()
		
	return int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], open(sys.argv[6], 'w'), open(sys.argv[7], 'w')

if __name__ == "__main__":
	read_length, reference_path, hom_table_path, read_path, cora_output_path, result_metadata, result_fq = parse_args()
	
	call_cora(read_length, reference_path, hom_table_path, read_path, cora_output_path)
	reconstruct_md_tags(cora_output_path, reference_path)
	filter_file(cora_output_path, result_metadata, result_fq)
	
