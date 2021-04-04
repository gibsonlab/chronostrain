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
# =====================================

# =========== Run metaphlan. ==================
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR
mkdir -p sams
mkdir -p bowtie2
rm bowtie2/*
mkdir -p profiles

for f in $READS_DIR/*.fastq
do
	bn=$(basename ${f%.fastq})
	echo "Running metaphlan on ${f} (basename=${bn})"

	metaphlan $f \
	--input_type fastq \
	-s sams/${bn}.sam.bz2 \
	--bowtie2out bowtie2/${bn}.bowtie2.bz2 \
	-o profiles/${bn}_profile.tsv \
	--index "mpa_chronostrain" \
	--bowtie2db "${CHRONOSTRAIN_DATA_DIR}/metaphlan_db"
done
# ================================================
