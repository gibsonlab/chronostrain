#!/bin/bash
set -e
source settings.sh

participant=$1
require_variable 'participant' $participant
require_file ${CHRONOSTRAIN_DB_JSON}

# =========== Run chronostrain. ==================
run_dir=${DATA_DIR}/${participant}/chronostrain
export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/filter.log"
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/.cache"
cd ${BASE_DIR}


echo "[*] Creating read input file."
append_fastq()
{
	gzip_fq_path=$1
	time=$2
	read_type=$3
	qual_fmt=$4
	sample_name=$5

	num_lines=$(pigz -dc $gzip_fq_path | wc -l | awk '{print $1}')
	num_reads=$((${num_lines} / 4))

	if [[ -s "${gzip_fq_path}" ]] && [[ ${num_reads} > 0 ]]; then
		echo "Adding record ${gzip_fq_path} to ${run_dir}/reads.csv"
		echo "${time},${sample_name},${num_reads},\"${gzip_fq_path}\",${read_type},${qual_fmt}" >> ${run_dir}/reads.csv
	else
		echo "Skipping empty record ${gzip_fq_path}"
	fi
}

echo "[**] Generating read CSV file ${run_dir}/reads.csv"
mkdir -p ${run_dir}
> ${run_dir}/reads.csv

while IFS=$'\t' read -r p_name time_point sample_id fq1_rel fq2_rel
do
  fq1=${DATA_DIR}/${participant}/reads/${sample_id}_1.fastq.gz
  fq2=${DATA_DIR}/${participant}/reads/${sample_id}_2.fastq.gz
  append_fastq "${fq1}" "$time_point" "paired_1" "fastq" "${sample_id}_PAIRED"
  append_fastq "${fq2}" "$time_point" "paired_2" "fastq" "${sample_id}_PAIRED"
done < ${DATA_DIR}/${participant}/dataset.tsv


echo "[*] Running filter on participant ${participant}."
env JAX_PLATFORM_NAME=cpu chronostrain filter \
  -r ${run_dir}/reads.csv \
  -o ${run_dir}/filtered \
  -s ${CHRONOSTRAIN_CLUSTER_FILE} \
  --aligner "bwa-mem2"
touch ${run_dir}/filtered/FILTER_DONE.txt