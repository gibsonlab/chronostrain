#!/bin/bash
set -e

#if [ -z ${PROJECT_DIR} ]; then
#	echo "Variable 'PROJECT_DIR' is not set. Exiting."
#	exit 1
#else
#	echo "PROJECT_DIR=${PROJECT_DIR}"
#fi

PROJECT_DIR="/PHShome/yk847/chronostrain"
CHRONOSTRAIN_DATA_DIR="/data/cctm/chronostrain"

# =====================================
# Command line args
NUM_READS=$1
TRIAL=$2

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

RUNS_DIR="${CHRONOSTRAIN_DATA_DIR}/runs"
READ_LEN=150
TRIAL_DIR="${RUNS_DIR}/trials/reads_${NUM_READS}_trial_${TRIAL}"

READS_DIR="${TRIAL_DIR}/simulated_reads"
OUTPUT_DIR="${TRIAL_DIR}/output/metaphlan"
METAPHLAN_DB="${CHRONOSTRAIN_DATA_DIR}/metaphlan_db"
# =====================================

# =========== Run metaphlan. ==================
mkdir -p $OUTPUT_DIR

echo "Output dir: ${OUTPUT_DIR}"
cd $OUTPUT_DIR

mkdir -p sams
mkdir -p bowtie2
mkdir -p profiles

# Cleanup previous bowtie2 output (metaphlan with --bowtie2out crashes prematurely otherwise).
bowtie2_out_format=bowtie2/*.bowtie2.bz2
for f in $bowtie2_out_format;
do
	echo $f
	if [[ "$f" != "$bowtie2_out_format" ]]
	then
		rm ${f}
	fi
done

# Loop through reads and run metaphlan.
for f in $READS_DIR/*.fastq
do
	echo $f
	bn=$(basename ${f%.fq})
	echo "Running metaphlan on: ${f} (basename=${bn})"
	echo "Using metaphlan database: ${METAPHLAN_DB}"

	metaphlan $f \
	--input_type fastq \
	-s sams/${bn}.sam.bz2 \
	--bowtie2out bowtie2/${bn}.bowtie2.bz2 \
	-o profiles/${bn}_profile.tsv \
	--bowtie2db ${METAPHLAN_DB} \
	--index "mpa_v30_CHOCOPhlAn_201901"

done
# ================================================
