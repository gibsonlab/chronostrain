#!/bin/bash
source settings.sh
check_program 'kneaddata'
check_program 'trimmomatic'
check_program 'pigz'
check_program 'date'

patient_subdir()
{
	target_patient="$1"
	echo "${SAMPLES_DIR}/${target_patient}"
}


datediff()
{
	# Return the number of days between two dates (provided in increasing order)
	date1=$(date +%s --date "$1")
	date2=$(date +%s --date "$2")
	diff=$(( (date2 - date1) / 86400 ))
	echo "$diff"
}

append_fastq()
{
	gzip_fq_path=$1
	time=$2
	read_type=$3
	qual_fmt=$4
	chronostrain_index_file=$5

	num_lines=$(pigz -dc $gzip_fq_path | wc -l | awk '{print $1}')
	num_reads=$((${num_lines} / 4))

	if [[ -s "${gzip_fq_path}" ]] && [[ ${num_reads} > 0 ]]; then
		echo "Adding record ${gzip_fq_path} to ${chronostrain_index_file}"
		echo "${time},${num_reads},\"${gzip_fq_path}\",${read_type},${qual_fmt}" >> "${chronostrain_index_file}"
	else
		echo "Skipping empty record ${gzip_fq_path}"
	fi
}


target_patient=$1
sample_subdir=$(patient_subdir "$target_patient" | tr -d '\r')
sample_index_path="${sample_subdir}/index.csv"

EPOCH="01/01/2013"

chronostrain_index_file="${READS_DIR}/${target_patient}/inputs.csv"
# Delete if exists
if [[ -f ${chronostrain_index_file} ]]; then rm "${chronostrain_index_file}"; fi
mkdir -p ${READS_DIR}

while IFS=, read -r project_id date_of_receipt; do
	gz_file_1="${sample_subdir}/${project_id}_1.fastq.gz"
	gz_file_2="${sample_subdir}/${project_id}_2.fastq.gz"
	days=$(datediff "$EPOCH" "$date_of_receipt")

	echo "[*] -=-=-=-=-=-=-=-= Handling ${project_id} (${date_of_receipt} of ${target_patient}). =-=-=-=-=-=-=-=-"

	# Target fastq files.
	kneaddata_output_dir="${sample_subdir}/kneaddata/${project_id}"
	mkdir -p $kneaddata_output_dir

	trimmed_1_paired_gz="${kneaddata_output_dir}/${project_id}_paired_1.fastq.gz"
	trimmed_1_unpaired_gz="${kneaddata_output_dir}/${project_id}_unmatched_1.fastq.gz"
	trimmed_2_paired_gz="${kneaddata_output_dir}/${project_id}_paired_2.fastq.gz"
	trimmed_2_unpaired_gz="${kneaddata_output_dir}/${project_id}_unmatched_2.fastq.gz"

	if [ -f "${trimmed_1_paired_gz}" ] && [ -f "${trimmed_2_paired_gz}" ]
	then
		echo "[*] Processed outputs already found!"
	else
		tmp_file_1="${kneaddata_output_dir}/${project_id}_1.fastq"
		tmp_file_2="${kneaddata_output_dir}/${project_id}_2.fastq"
		echo "[*] Decompressing to temporary output."
		pigz -dck ${gz_file_1} > ${tmp_file_1}
		pigz -dck ${gz_file_2} > ${tmp_file_2}

		echo "[*] Invoking kneaddata."
		kneaddata \
		--input1 ${tmp_file_1} \
		--input2 ${tmp_file_2} \
		--reference-db ${KNEADDATA_DB_DIR} \
		--output ${kneaddata_output_dir} \
		--trimmomatic-options "SLIDINGWINDOW:100:0 MINLEN:35 ILLUMINACLIP:${NEXTERA_ADAPTER_PATH}:2:40:15" \
		--threads 6 \
		--quality-scores phred33 \
		--bypass-trf \
		--trimmomatic ${TRIMMOMATIC_DIR} \
		--output-prefix ${project_id}

		echo "[*] Compressing fastq files."
		trimmed_1_paired="${kneaddata_output_dir}/${project_id}_paired_1.fastq"
		trimmed_1_unpaired="${kneaddata_output_dir}/${project_id}_unmatched_1.fastq"
		trimmed_2_paired="${kneaddata_output_dir}/${project_id}_paired_2.fastq"
		trimmed_2_unpaired="${kneaddata_output_dir}/${project_id}_unmatched_2.fastq"
		pigz $trimmed_1_paired
		pigz $trimmed_1_unpaired
		pigz $trimmed_2_paired
		pigz $trimmed_2_unpaired

		echo "[*] Cleaning up..."
		for f in ${kneaddata_output_dir}/*.fastq; do rm $f; done
	fi

	# Add to timeseries input index.
	append_fastq "${trimmed_1_paired_gz}" "$days" "paired_1" "fastq" "${chronostrain_index_file}"
	append_fastq "${trimmed_1_unpaired_gz}" "$days" "paired_1" "fastq" "${chronostrain_index_file}"
	append_fastq "${trimmed_2_paired_gz}" "$days" "paired_2" "fastq" "${chronostrain_index_file}"
	append_fastq "${trimmed_2_unpaired_gz}" "$days" "paired_2" "fastq" "${chronostrain_index_file}"
done < ${sample_index_path}
