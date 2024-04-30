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
subdir=$6
n_iters=$7
score=$8

require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial
require_variable "time_point" $time_point
require_variable "subdir" $subdir
require_variable "n_iters" $n_iters
require_variable "score" $score


trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/${subdir}


# ========== Run
runtime_file=${output_dir}/runtime.${time_point}.txt
if [[ -f $runtime_file ]]; then
	echo "[*] Skipping StrainGST run (replicate: ${replicate} | n_reads: ${n_reads} | trial: ${trial} | timepoint #${time_point})"
	exit 0
fi

echo "[*] Running inference for (replicate: ${replicate} | n_reads: ${n_reads} | trial: ${trial} | timepoint #${time_point})"
start_time=$(date +%s%N)  # nanoseconds
read_kmers=${trial_dir}/output/straingst_kmers/reads.${time_point}.hdf5
mkdir -p ${output_dir}

if [ -f ${read_kmers} ]; then
  echo "[*] Skipping k-merization."
else
  bash parameter_sensitivity/strainge_kmerize.sh $mutation_ratio $replicate $n_reads $trial $time_point
fi

echo "[*] Running StrainGST (output subdir=${subdir})"
mkdir -p ${output_dir}
straingst run \
-o ${output_dir}/output_${time_point}.tsv \
-i ${n_iters} \
${STRAINGE_DB_DIR}/database.hdf5 \
${read_kmers} \
-s ${score} \
--separate-output


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file
