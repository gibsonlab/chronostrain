#!/bin/bash
set -e
source settings.sh
source strainge/settings.sh

# ============ Requires arguments:
mutation_ratio=$1
replicate=$2
n_reads=$3
trial=$4
time_point=$5

require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial
require_variable "time_point" $time_point




trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/straingst


# ========== Run
runtime_file=${trial_dir}/output/straingst_runtime.${time_point}.txt
if [[ -f $runtime_file ]]; then
	echo "[*] Skipping StrainGST run (replicate: ${replicate} | n_reads: ${n_reads} | trial: ${trial} | timepoint #${time_point})"
	exit 0
fi

echo "[*] Running inference for (replicate: ${replicate} | n_reads: ${n_reads} | trial: ${trial} | timepoint #${time_point})"
start_time=$(date +%s%N)  # nanoseconds
read_kmers=${output_dir}/reads.${time_point}.hdf5
mkdir -p ${output_dir}


echo "[*] Kmerizing..."
straingst kmerize \
-k 23 \
-o ${read_kmers} \
${read_dir}/${time_point}_sim_1.fq \
${BACKGROUND_FASTQ_DIR}/${time_point}_background_1.fq \
${read_dir}/${time_point}_sim_2.fq \
${BACKGROUND_FASTQ_DIR}/${time_point}_background_2.fq


echo "[*] Running StrainGST."
mkdir -p ${output_dir}
straingst run \
-o ${output_dir}/output_${time_point}.tsv \
-i 10 \
${STRAINGE_DB_DIR}/database.hdf5 \
${read_kmers} \
--separate-output \
-s 0.0


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file


echo "[*] Cleaning up."
rm ${read_kmers}
