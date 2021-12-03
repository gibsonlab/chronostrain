#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
# REQUIRES: sratools (prefetch + fasterq-dump)
check_program 'prefetch'
check_program 'fasterq-dump'

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
	while IFS=, read -r sra_id umb_id sample_name date days experiment_type
	do
		if [[ "${experiment_type}" != "stool" ]]; then
			echo "Skipping ${sample_name}."
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

		# Obtained fastq files.
		fq_file_1="${SAMPLES_DIR}/${sra_id}_1.fastq"
		fq_file_2="${SAMPLES_DIR}/${sra_id}_2.fastq"

		echo "[*] Compressing..."
		gzip $fq_file_1 --force
		gzip $fq_file_2 --force
	done
} < ${SRA_CSV_PATH}
