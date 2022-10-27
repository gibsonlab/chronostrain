export NCBI_REFSEQ_DIR=./ncbi-genomes
export NUM_CORES=4
export INDEX_FILE=${NCBI_REFSEQ_DIR}/index.tsv


check_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}
export check_program

