#!/bin/bash
set -e
source settings.sh

# ============ Requires arguments:
n_reads=$1
trial=$2

if [ -z "$n_reads" ]
then
	echo "var \"n_reads\" is empty"
	exit 1
fi

if [ -z "$trial" ]
then
	echo "var \"trial\" is empty"
	exit 1
fi

# ============ script body:
trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/chronostrain

mkdir -p $output_dir
mkdir -p $read_dir
export CHRONOSTRAIN_LOG_FILEPATH="${output_dir}/filter.log"
export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"


echo "[*] Preparing input for n_reads: ${n_reads}, trial: ${trial}"
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


echo "[*] Filtering reads..."
start_time=$(date +%s%N)  # nanoseconds

chronostrain filter \
-r ${reads_csv} \
-o "${output_dir}/filtered"

# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
mkdir -p ${trial_dir}/output
runtime_file=${trial_dir}/output/chronostrain_filter_runtime.txt
echo "${elapsed_time}" > $runtime_file
