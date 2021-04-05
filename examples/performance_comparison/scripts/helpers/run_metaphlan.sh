#!/bin/bash
set -e

# =============================
# Script purpose: Run metaphlan on specified input.
#
# This script assumes variables READS_DIR, OUTPUT_DIR, METAPHLAN_DB, METAPHLAN_DB_INDEX.
# =============================

# Setup directories.
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
	echo "Using metaphlan database: ${METAPHLAN_DB}, index ${METAPHLAN_DB_INDEX}"

	metaphlan $f \
	--input_type fastq \
	-s sams/${bn}.sam.bz2 \
	--bowtie2out bowtie2/${bn}.bowtie2.bz2 \
	-o profiles/${bn}_profile.tsv \
	--bowtie2db ${METAPHLAN_DB} \
	--index ${METAPHLAN_DB_INDEX}
done
# ================================================
