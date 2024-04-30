#!/bin/bash
set -e
source settings.sh

participant=$1
mutation_rate=$2
require_variable 'participant' $participant
require_variable 'mutation_rate' $mutation_rate
require_file ${CHRONOSTRAIN_DB_JSON}

# =========== Run chronostrain. ==================
echo "[*] Running filter on participant ${participant}, with mutated db p=0.${mutation_rate}."

run_dir=${DATA_DIR}/${participant}/chronostrain_mutation_${mutation_rate}
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
  if [ "${sample_id}" == "SampleId" ]; then continue; fi
  fq1=${DATA_DIR}/${participant}/reads/${sample_id}_1.fastq.gz
  fq2=${DATA_DIR}/${participant}/reads/${sample_id}_2.fastq.gz
  append_fastq "${fq1}" "$time_point" "paired_1" "fastq" "${sample_id}_PAIRED"
  append_fastq "${fq2}" "$time_point" "paired_2" "fastq" "${sample_id}_PAIRED"
done < ${DATA_DIR}/${participant}/dataset.tsv


export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/filter.log"
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/.cache_mut_${mutation_rate}"
export CHRONOSTRAIN_DB_JSON=${DATA_DIR}/database/mutated_dbs/${mutation_rate}/chronostrain/efaecalis.json
export CHRONOSTRAIN_CLUSTERS=${DATA_DIR}/database/mutated_dbs/${mutation_rate}/chronostrain/efaecalis.clusters.txt

env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu \
  chronostrain filter \
  -r ${run_dir}/reads.csv \
  -o ${run_dir}/filtered \
  -s ${CHRONOSTRAIN_CLUSTERS} \
  --aligner "bwa-mem2"

env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu \
  chronostrain precompute \
  -r ${run_dir}/filtered/filtered_reads.csv \
  -s ${CHRONOSTRAIN_CLUSTERS}
