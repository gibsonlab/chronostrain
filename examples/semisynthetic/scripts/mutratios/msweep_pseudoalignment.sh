#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh

# This script is the same as the basic run_pseudoalignment.sh, but it excludes the background reads. (Those will be included for mSWEEP itself)

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
themisto_db_dir="${DATA_DIR}/databases/themisto"
trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/themisto
runtime_file=${output_dir}/runtime.txt


if [ -f $runtime_file ]; then
	echo "[*] Skipping Themisto Pseudoalignment (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
	exit 0
fi


echo "[*] Running Themisto pseudoalignment (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
mkdir -p ${output_dir}


echo "[**] Preparing themisto inputs..."
input_file=${output_dir}/query_files.txt
output_file=${output_dir}/output_files.txt
> $input_file
> $output_file

for t_idx in 0 1 2 3 4 5; do
  echo "${read_dir}/${t_idx}_sim_1.fq" >> $input_file
  echo "${read_dir}/${t_idx}_sim_2.fq" >> $input_file
  echo "${output_dir}/${t_idx}_sim_1.output.txt" >> $output_file
  echo "${output_dir}/${t_idx}_sim_2.output.txt" >> $output_file
done


echo "[**] Running pseudoalignment (target database: ${themisto_db_dir})"
start_time=$(date +%s%N)  # nanoseconds
tmp_dir="${output_dir}/_tmp"
${THEMISTO_BIN_DIR}/themisto pseudoalign \
  --query-file-list $input_file \
  --out-file-list $output_file \
  --index-prefix "${themisto_db_dir}/enterobacteriaceae" \
  --temp-dir $tmp_dir \
  --n-threads $N_CORES \
  --sort-output


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file


# clean up.
echo '[**] Cleaning up...'
rm -rf $tmp_dir