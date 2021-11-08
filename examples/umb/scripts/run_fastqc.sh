#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
check_program 'fastqc'

# ================================= Main script ==================================

# Clear index file.
cd ${SAMPLES_DIR}
mkdir fastqc

for fqfile in ./*.fastq; do
	basename=${fqfile%.fastq}
	basename=${basename##*/}
	outdir="fastqc/${basename}"
	echo "-=-=-=-=-=-=-=-= Handling ${fqfile} =-=-=-=-=-=-=-=-"
	mkdir ${outdir}
	fastqc ${fqfile} -o ${outdir} -f fastq
done
