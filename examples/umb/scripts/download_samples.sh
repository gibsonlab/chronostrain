#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
# REQUIRES: sratools (prefetch + fasterq-dump), trimmomatic, gzip
check_program 'gzip'
check_program 'prefetch'
check_program 'fasterq-dump'
check_program 'trimmomatic'

# Gzips the input fastq file, and appends the fastq-timepoint pair as an entry.
gzip_and_append_fastq()
{
	fq_path=$1
	time=$2
	umb_id=$3

	num_lines=$(wc -l $fq_path | awk '{print $1}')
	num_reads=$((${num_lines} / 4))

	if [[ -s "${fq_path}" ]]; then
  	echo "[*] Compressing: ${fq_path}"
		gzip $fq_path --force
		gz_file="${fq_path}.gz"
		echo "\"${time}\",\"${num_reads}\",\"${gz_file}\"" >> "${READS_DIR}/${umb_id}_${INPUT_INDEX_FILENAME}"
	else
  	echo "[*] Skipping empty file ${fq_path}."
	fi

	echo "[*] Cleaning up ${fq_path}."
	rm ${fq_path}
}

# ================================= Main script ==================================

# Clear index file.
mkdir -p ${READS_DIR}
touch $INPUT_INDEX_PATH
> $INPUT_INDEX_PATH

SRA_CSV_PATH="${BASE_DIR}/files/umb_samples.csv"

mkdir -p ${SAMPLES_DIR}
mkdir -p ${SRA_PREFETCH_DIR}
mkdir -p "${SAMPLES_DIR}/trimmomatic"
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

		echo "-=-=-=-=-=-=-=-= Handling ${sra_id} (${sample_name}). =-=-=-=-=-=-=-=-"

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

		# Target fastq files.
		trimmed_paired_1="${SAMPLES_DIR}/trimmomatic/${sra_id}_1_paired.fastq"
		trimmed_unpaired_1="${SAMPLES_DIR}/trimmomatic/${sra_id}_1_unpaired.fastq"
		trimmed_paired_2="${SAMPLES_DIR}/trimmomatic/${sra_id}_2_paired.fastq"
		trimmed_unpaired_2="${SAMPLES_DIR}/trimmomatic/${sra_id}_2_unpaired.fastq"

		# Preprocess
		echo "[*] Invoking trimmomatic..."
		trimmomatic PE \
		-threads 4 \
		-phred33 \
		${fq_file_1} ${fq_file_2} \
		${trimmed_paired_1} ${trimmed_unpaired_1} \
		${trimmed_paired_2} ${trimmed_unpaired_2} \
		SLIDINGWINDOW:100:0 \
		MINLEN:35 \
		ILLUMINACLIP:${NEXTERA_ADAPTER_PATH}:2:40:15

		# Add to timeseries input index.
		gzip_and_append_fastq ${trimmed_paired_1} $days $umb_id
		gzip_and_append_fastq ${trimmed_unpaired_1} $days $umb_id
	done
} < ${SRA_CSV_PATH}
