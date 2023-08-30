#!/bin/bash
set -e
source settings.sh

# ============ Requires arguments:
mutation_ratio=$1
replicate=$2
n_reads=$3
trial=$4

require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial

# ============ script body:
replicate_dir=$(get_replicate_dir "${mutation_ratio}" "${replicate}")
trial_dir=$(get_trial_dir "${mutation_ratio}" $replicate $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/chronostrain
runtime_file=${trial_dir}/output/chronostrain_filter_runtime.txt

mkdir -p $output_dir
mkdir -p $read_dir


if [[ -f $runtime_file ]]; then
	echo "[*] Skipping Filter (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
	exit 0
fi


echo "[*] Preparing input for (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
reads_csv="${output_dir}/input_files.csv"
>$reads_csv
{
	read
	while IFS=, read -r tidx t sra n_background
	do
		echo "${t},${n_reads},${read_dir}/${tidx}_sim_1.fq,paired_1,fastq" >> $reads_csv
		echo "${t},${n_reads},${read_dir}/${tidx}_sim_2.fq,paired_2,fastq" >> $reads_csv
		echo "${t},${n_background},${BACKGROUND_FASTQ_DIR}/${tidx}_background_1.fq,paired_1,fastq" >> $reads_csv
		echo "${t},${n_background},${BACKGROUND_FASTQ_DIR}/${tidx}_background_2.fq,paired_2,fastq" >> $reads_csv
	done
} < $BACKGROUND_CSV


echo "[*] Filtering reads (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
start_time=$(date +%s%N)  # nanoseconds

env JAX_PLATFORM_NAME=cpu \
  CHRONOSTRAIN_DB_JSON=${replicate_dir}/databases/chronostrain/ecoli.json \
  CHRONOSTRAIN_DB_DIR=${replicate_dir}/databases/chronostrain \
  CHRONOSTRAIN_LOG_FILEPATH=${output_dir}/filter.log \
  CHRONOSTRAIN_CACHE_DIR=${output_dir}/cache \
  chronostrain filter \
  -r ${reads_csv} \
  -o "${output_dir}/filtered" \
  --aligner bwa-mem2

# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
mkdir -p ${trial_dir}/output
echo "${elapsed_time}" > $runtime_file
