#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
# REQUIRES: kneaddata, trimmomatic, gzip
require_program 'kneaddata'
require_program 'trimmomatic'
require_program 'pigz'

# Gzips the input fastq file, and appends the fastq-timepoint pair as an entry.
append_fastq()
{
	gzip_fq_path=$1
	time=$2
	umb_id=$3
	read_type=$4
	qual_fmt=$5
	sample_name=$6

	num_lines=$(pigz -dc $gzip_fq_path | wc -l | awk '{print $1}')
	num_reads=$((${num_lines} / 4))
  mkdir -p ${OUTPUT_DIR}/${umb_id}
  reads_file=${OUTPUT_DIR}/${umb_id}/reads.csv

	if [[ -s "${gzip_fq_path}" ]] && [[ ${num_reads} > 0 ]]; then
		echo "Adding record ${gzip_fq_path} to ${reads_file}"
		echo "${time},${sample_name},${num_reads},\"${gzip_fq_path}\",${read_type},${qual_fmt}" >> ${reads_file}
	else
		echo "Skipping empty record ${gzip_fq_path}"
	fi
}

# ================================= Main script ==================================

# Clear index file.
mkdir -p ${OUTPUT_DIR}
find ${OUTPUT_DIR} -maxdepth 2 -name reads.csv -type f -exec rm '{}' \;

SRA_CSV_PATH="${BASE_DIR}/files/umb_samples.csv"

mkdir -p ${SAMPLES_DIR}
mkdir -p ${SRA_PREFETCH_DIR}
mkdir -p "${SAMPLES_DIR}/kneaddata"

# ================== Parse CSV file.
{
	# Skip header line.
	read

	# Read rest of csv file.
	while IFS=, read -r sra_id umb_id umb_sample_name date days experiment_type model library_strategy exp_group
	do
		if [[ "${experiment_type}" != "stool" ]]; then
			echo "Skipping ${umb_sample_name}."
			continue
		fi

#		if [[ "${exp_group}" != "Test" ]]; then
#			echo "Skipping ${umb_sample_name}. (is not test group)"
#			continue
#		fi

		echo "[*] -=-=-=-=-=-=-=-= Handling ${sra_id} (${umb_sample_name}). =-=-=-=-=-=-=-=-"

		# Obtained fastq files.
		fq_file_1="${SAMPLES_DIR}/${sra_id}_1.fastq.gz"
		fq_file_2="${SAMPLES_DIR}/${sra_id}_2.fastq.gz"

		# Target fastq files.
		kneaddata_output_dir="${SAMPLES_DIR}/kneaddata/${sra_id}"
		mkdir -p $kneaddata_output_dir

		trimmed_1_paired_gz="${kneaddata_output_dir}/${sra_id}_paired_1.fastq.gz"
		trimmed_1_unpaired_gz="${kneaddata_output_dir}/${sra_id}_unmatched_1.fastq.gz"
		trimmed_2_paired_gz="${kneaddata_output_dir}/${sra_id}_paired_2.fastq.gz"
		trimmed_2_unpaired_gz="${kneaddata_output_dir}/${sra_id}_unmatched_2.fastq.gz"

		if [ -f "${trimmed_1_paired_gz}" ] && [ -f "${trimmed_2_paired_gz}" ]
		then
			echo "[*] Processed outputs already found!"
		else
			tmp_file_1="${kneaddata_output_dir}/${sra_id}_1.fastq"
			tmp_file_2="${kneaddata_output_dir}/${sra_id}_2.fastq"
			echo "[*] Decompressing to temporary output."
			pigz -dck ${fq_file_1} > $tmp_file_1
			pigz -dck ${fq_file_2} > $tmp_file_2

			echo "[*] Invoking kneaddata."
			kneaddata \
			--input1 ${tmp_file_1} \
			--input2 ${tmp_file_2} \
			--reference-db ${KNEADDATA_DB_DIR} \
			--output ${kneaddata_output_dir} \
			--trimmomatic-options "ILLUMINACLIP:${NEXTERA_ADAPTER_PATH}:2:30:10:2 LEADING:10 TRAILING:10 MINLEN:35" \
			--threads 8 \
			--quality-scores phred33 \
			--bypass-trf \
			--trimmomatic ${TRIMMOMATIC_DIR} \
			--output-prefix ${sra_id}

			echo "[*] Compressing fastq files."
			trimmed_1_paired="${kneaddata_output_dir}/${sra_id}_paired_1.fastq"
			trimmed_1_unpaired="${kneaddata_output_dir}/${sra_id}_unmatched_1.fastq"
			trimmed_2_paired="${kneaddata_output_dir}/${sra_id}_paired_2.fastq"
			trimmed_2_unpaired="${kneaddata_output_dir}/${sra_id}_unmatched_2.fastq"
			pigz $trimmed_1_paired
			pigz $trimmed_1_unpaired
			pigz $trimmed_2_paired
			pigz $trimmed_2_unpaired

			echo "[*] Cleaning up..."
			for f in ${kneaddata_output_dir}/*.fastq; do rm $f; done
		fi

		# Add to timeseries input index.
		append_fastq ${trimmed_1_paired_gz} $days $umb_id "paired_1" "fastq" "${sra_id}_PAIRED"
		append_fastq ${trimmed_1_unpaired_gz} $days $umb_id "paired_1" "fastq" "${sra_id}_UNPAIRED_1"
		append_fastq ${trimmed_2_paired_gz} $days $umb_id "paired_2" "fastq" "${sra_id}_PAIRED"
		append_fastq ${trimmed_2_unpaired_gz} $days $umb_id "paired_2" "fastq" "${sra_id}_UNPAIRED_2"
	done
} < ${SRA_CSV_PATH}
