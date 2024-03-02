#!/bin/bash
set -e
source settings.sh
source chronostrain/settings.sh
source mutratios/settings.sh

# Note: this script is effectively the same as the main "filter.sh" script, but assumes that the main analysis'
# filter.sh already ran.
#
# Using that assumption, this script ONLY filters the simulated reads -- and merges it with the background filtered
# read files that were already produced (those do not vary across replicates or read depths)

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
trial_dir=$(get_trial_dir "${mutation_ratio}" $replicate $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/chronostrain
runtime_file=${trial_dir}/output/chronostrain_filter_runtime.txt

mkdir -p "$output_dir"
mkdir -p "$read_dir"


if [[ -f $runtime_file ]]; then
	echo "[*] Skipping Filter (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
	exit 0
fi


echo "[*] Preparing input for (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
reads_csv="${output_dir}/input_files.csv"
if [ -f "$reads_csv" ]; then rm "$reads_csv"; fi
{
	read
	while IFS=, read -r tidx t sra n_background
	do
		echo "${t},SIM${tidx},${n_reads},${read_dir}/${tidx}_sim_1.fq,paired_1,fastq" >> "$reads_csv"
		echo "${t},SIM${tidx},${n_reads},${read_dir}/${tidx}_sim_2.fq,paired_2,fastq" >> "$reads_csv"
#		echo "${t},BACKGROUND${tidx},${n_background},${BACKGROUND_FASTQ_DIR}/${tidx}_background_1.fq,paired_1,fastq" >> "$reads_csv"
#		echo "${t},BACKGROUND${tidx},${n_background},${BACKGROUND_FASTQ_DIR}/${tidx}_background_2.fq,paired_2,fastq" >> "$reads_csv"
	done
} < "$BACKGROUND_CSV"


echo "[*] Filtering reads (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
start_time=$(date +%s%N)  # nanoseconds

env JAX_PLATFORM_NAME=cpu \
  CHRONOSTRAIN_DB_JSON=${CHRONOSTRAIN_DB_JSON_SRC} \
  CHRONOSTRAIN_DB_DIR=${DATA_DIR}/databases/chronostrain \
  CHRONOSTRAIN_LOG_FILEPATH=${output_dir}/filter.log \
  CHRONOSTRAIN_CACHE_DIR=${CHRONOSTRAIN_CACHE_DIR} \
  chronostrain filter \
  -r ${reads_csv} \
  -o "${output_dir}/filtered" \
  -s ${CHRONOSTRAIN_CLUSTER_FILE} \
  --aligner bwa-mem2


# Add the pre-filtered background reads. (saves some time; runtimes are not for benchmarking.)
{
	read
	while IFS=, read -r tidx t sra n_background
	do
		echo "${t},BACKGROUND${tidx},${n_background},/mnt/e/semisynthetic_data/mutratio_1.0/replicate_1/reads_10000/trial_1/output/chronostrain/filtered/filtered_${tidx}_background_1.fastq,paired_1,fastq" >> "${output_dir}/filtered/filtered_input_files.csv"
		echo "${t},BACKGROUND${tidx},${n_background},/mnt/e/semisynthetic_data/mutratio_1.0/replicate_1/reads_10000/trial_1/output/chronostrain/filtered/filtered_${tidx}_background_2.fastq,paired_2,fastq" >> "${output_dir}/filtered/filtered_input_files.csv"
	done
} < "$BACKGROUND_CSV"


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
mkdir -p ${trial_dir}/output
echo "${elapsed_time}" > $runtime_file
