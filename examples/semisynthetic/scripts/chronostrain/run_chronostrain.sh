#!/bin/bash
set -e
source settings.sh
source chronostrain/settings.sh

# ============ Requires arguments:
mutation_ratio=$1
replicate=$2
n_reads=$3
trial=$4
out_subdir=$5
prior_p=$6

require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial
require_variable "out_subdir" $out_subdir
require_variable "prior_p" $prior_p

# ============ script body:
trial_dir=$(get_trial_dir "${mutation_ratio}" $replicate $n_reads $trial)
output_dir=${trial_dir}/output/${out_subdir}
runtime_file=${output_dir}/inference_runtime.txt
filter_file=${output_dir}/filter_runtime.txt


if [ -f $runtime_file ]; then
	echo "[*] Skipping Chronostrain Inference (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
	exit 0
fi

if [ ! -f $filter_file ]; then
	echo "[*] Filtered result not found for (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
	exit 1
fi


mkdir -p $output_dir

echo "[*] Running Chronostrain inference (output subdir: ${out_subdir} | mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
echo "[**] Using database ${CHRONOSTRAIN_DB_JSON_SRC}"
start_time=$(date +%s%N)  # nanoseconds

env \
  CHRONOSTRAIN_DB_JSON=${CHRONOSTRAIN_DB_JSON_SRC} \
  CHRONOSTRAIN_DB_DIR=${DATA_DIR}/databases/chronostrain \
  CHRONOSTRAIN_LOG_FILEPATH=${output_dir}/inference.log \
  CHRONOSTRAIN_CACHE_DIR=${CHRONOSTRAIN_CACHE_DIR} \
  chronostrain advi \
  -r "${trial_dir}/output/chronostrain/filtered/filtered_input_files.csv" \
  -o ${output_dir} \
  -s ${CHRONOSTRAIN_CLUSTER_FILE} \
  --correlation-mode "full" \
  --iters ${CHRONOSTRAIN_NUM_ITERS} \
  --epochs ${CHRONOSTRAIN_NUM_EPOCHS} \
  --decay-lr ${CHRONOSTRAIN_DECAY_LR} \
  --lr-patience ${CHRONOSTRAIN_LR_PATIENCE} \
  --loss-tol ${CHRONOSTRAIN_LOSS_TOL} \
  --learning-rate ${CHRONOSTRAIN_LR} \
  --num-samples ${CHRONOSTRAIN_NUM_SAMPLES} \
  --read-batch-size ${CHRONOSTRAIN_READ_BATCH_SZ} \
  --min-lr ${CHRONOSTRAIN_MIN_LR} \
  --plot-format "pdf" \
  --plot-elbo \
  --prune-strains \
  --with-zeros \
  --prior-p ${prior_p}
#	--accumulate-gradients


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file
