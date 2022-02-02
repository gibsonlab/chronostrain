#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
# REQUIRES: kneaddata, trimmomatic, gzip
check_program 'kneaddata'
check_program 'trimmomatic'
check_program 'gzip'

# Gzips the input fastq file, and appends the fastq-timepoint pair as an entry.
append_fastq()
{
	gzip_fq_path=$1
	time=$2
	umb_id=$3

	num_lines=$(zcat $gzip_fq_path | wc -l | awk '{print $1}')
	num_reads=$((${num_lines} / 4))

	if [[ -s "${gzip_fq_path}" ]]; then
		echo "Adding record ${gzip_fq_path} to ${READS_DIR}/${umb_id}_${INPUT_INDEX_FILENAME}"
		echo "\"${time}\",\"${num_reads}\",\"${gzip_fq_path}\"" >> "${READS_DIR}/${umb_id}_${INPUT_INDEX_FILENAME}"
	else
		echo "Skipping empty record ${gzip_fq_path}"
	fi
}

# ================================= Main script ==================================

# Clear index file.
mkdir -p ${READS_DIR}

SRA_CSV_PATH="${BASE_DIR}/files/umb_samples.csv"

mkdir -p ${SAMPLES_DIR}
mkdir -p ${SRA_PREFETCH_DIR}
mkdir -p "${SAMPLES_DIR}/trimmomatic"

# ================== Parse CSV file.
{
	# Skip header line.
	read

	# Read rest of csv file.
	while IFS=, read -r sra_id umb_id sample_name date days experiment_type model library_strategy exp_group
	do
		if [[ "${experiment_type}" != "stool" ]]; then
			echo "Skipping ${sample_name}."
			continue
		fi

		if [[ "${exp_group}" != "Test" ]]; then
			echo "Skipping ${sample_name}. (is not test group)"
			continue
		fi

		echo "[*] -=-=-=-=-=-=-=-= Handling ${sra_id} (${sample_name}). =-=-=-=-=-=-=-=-"

		# Obtained fastq files.
		fq_file_1="${SAMPLES_DIR}/${sra_id}_1.fastq.gz"
		fq_file_2="${SAMPLES_DIR}/${sra_id}_2.fastq.gz"

		# Target fastq files.
		kneaddata_output_dir="${SAMPLES_DIR}/kneaddata/${sra_id}"
		trimmed_1_paired_gz="${kneaddata_output_dir}/${sra_id}_kneaddata_paired_1.fastq.gz"
		trimmed_1_unpaired_gz="${kneaddata_output_dir}/${sra_id}_kneaddata_unmatched_1.fastq.gz"
		trimmed_2_paired_gz="${kneaddata_output_dir}/${sra_id}_kneaddata_paired_2.fastq.gz"
		trimmed_2_unpaired_gz="${kneaddata_output_dir}/${sra_id}_kneaddata_unmatched_2.fastq.gz"

		if [ -f "${trimmed_1_paired_gz}" ] && [ -f "${trimmed_2_paired_gz}" ]
		then
			echo "Processed outputs already found!"
		else
			echo "[*] Invoking kneaddata..."
			kneaddata \
			--input1 ${fq_file_1} \
			--input2 ${fq_file_2} \
			--reference-db ${KNEADDATA_DB_DIR} \
			--output ${kneaddata_output_dir} \
			--trimmomatic-options "SLIDINGWINDOW:100:0 MINLEN:35 ILLUMINACLIP:${NEXTERA_ADAPTER_PATH}:2:40:15" \
			--threads 6 \
			--quality-scores phred33 \
			--bypass-trf \
			--trimmomatic ${TRIMMOMATIC_DIR} \
			--output-prefix ${sra_id}

			trimmed_1_paired="${kneaddata_output_dir}/${sra_id}_kneaddata_paired_1.fastq"
			trimmed_1_unpaired="${kneaddata_output_dir}/${sra_id}_kneaddata_unmatched_1.fastq"
			trimmed_2_paired="${kneaddata_output_dir}/${sra_id}_kneaddata_paired_2.fastq"
			trimmed_2_unpaired="${kneaddata_output_dir}/${sra_id}_kneaddata_unmatched_2.fastq"
			gzip trimmed_1_paired
			gzip trimmed_1_unpaired
			gzip trimmed_2_paired
			gzip trimmed_2_unpaired
		fi

		# Add to timeseries input index.
		append_fastq ${trimmed_1_paired_gz} $days $umb_id
		append_fastq ${trimmed_1_unpaired_gz} $days $umb_id
		append_fastq ${trimmed_2_paired_gz} $days $umb_id
		append_fastq ${trimmed_2_unpaired_gz} $days $umb_id
	done
} < ${SRA_CSV_PATH}
