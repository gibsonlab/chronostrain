#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
# REQUIRES: sratools (prefetch + fasterq-dump), pigz
check_program 'prefetch'
check_program 'fasterq-dump'
check_program 'pigz'

# ================================= Main script ==================================

SRA_CSV_PATH="${BASE_DIR}/files/umb_samples.csv"

mkdir -p ${SAMPLES_DIR}
mkdir -p ${SRA_PREFETCH_DIR}
mkdir -p ${FASTERQ_TMP_DIR}

# ================== Parse CSV file.
{
	# Skip header line.
	read

	# Read rest of csv file.
	while IFS=, read -r sra_id umb_id sample_name date days experiment_type model library_strategy exp_group
	do
		if [[ "${experiment_type}" != "urine raw" ]] && [[ "${experiment_type}" != "urine outgrowth" ]]; then
			echo "Skipping ${sample_name}."
			continue
		fi

		# Target fastq files.
		gz_file_1="${SAMPLES_DIR}/${sra_id}_1.fastq.gz"
		gz_file_2="${SAMPLES_DIR}/${sra_id}_2.fastq.gz"
		if [[ -f $gz_file_1 && -f $gz_file_2 ]]; then
			echo "[*] Target files for ${sra_id} already exist."
			continue
		fi

		echo "[*] -=-=-=-= Downloading ${sra_id} (${sample_name}) =-=-=-=-"

		# Prefetch
		echo "[*] Prefetching..."
		prefetch --output-directory $SRA_PREFETCH_DIR --progress --verify yes $sra_id

		# Fasterq-dump
		echo "[*] Invoking fasterq-dump..."
		fasterq-dump \
		--progress \
		--outdir $SAMPLES_DIR \
		--skip-technical \
		--print-read-nr \
		--force \
		-t ${FASTERQ_TMP_DIR} \
		"${SRA_PREFETCH_DIR}/${sra_id}/${sra_id}.sra"

		echo "[*] Compressing..."
		fq_file_1="${SAMPLES_DIR}/${sra_id}_1.fastq"
		fq_file_2="${SAMPLES_DIR}/${sra_id}_2.fastq"
		pigz $fq_file_1
		pigz $fq_file_2
		wait
	done
} < ${SRA_CSV_PATH}
