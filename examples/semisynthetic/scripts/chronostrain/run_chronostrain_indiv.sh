#!/bin/bash
# A variant of run_chronostrain.sh, which runs ChronoStrain on each sample separately without timeseries information.
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

out_subdir="chronostrain_indiv"  # debug, override dev-only

require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial
require_variable "out_subdir" $out_subdir
require_variable "prior_p" $prior_p

# ============ script body:
trial_dir=$(get_trial_dir "${mutation_ratio}" $replicate $n_reads $trial)

original_filter_subdir=${trial_dir}/output/chronostrain/filtered
filter_csv="${original_filter_subdir}/filtered_input_files.csv"
if [ ! -f $filter_csv ]; then
	echo "[*] Filtered result not found for (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
	exit 1
fi

echo "[*] Running Chronostrain inference (output subdir: ${out_subdir} | mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
echo "[**] Using database ${CHRONOSTRAIN_DB_JSON_SRC}"

# Iterate over all unique timepoints
t_idx=0
output_dir=${trial_dir}/output/${out_subdir}
mkdir -p $output_dir

for time_point in `cat ${filter_csv} | awk -F ',' '{print $1}' | uniq | sort -n`; do
  echo "[*] Running analysis on timepoint slice t=$time_point (index ${t_idx})"

  output_slice_dir="$output_dir/timepoint_${t_idx}"
  runtime_file=${output_slice_dir}/inference_runtime.txt
  if [ -f $runtime_file ]; then
    echo "[*] Skipping Chronostrain_indiv Inference (Timepoint: {t_idx} | mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
    continue
  fi

  mkdir -p "$output_slice_dir"
  # gather all samples with this timepoint, replace it with "100.0"
  cat ${filter_csv} | grep ^${time_point} | awk -v dir="${original_filter_subdir}" 'BEGIN{FS=OFS=","} {gsub("\"", "")} $1="100.0",$4=dir"/"$4' > "${output_slice_dir}/filtered_input.csv"
  start_time=$(date +%s%N)  # nanoseconds

  env \
    CHRONOSTRAIN_DB_JSON=${CHRONOSTRAIN_DB_JSON_SRC} \
    CHRONOSTRAIN_DB_DIR=${DATA_DIR}/databases/chronostrain \
    CHRONOSTRAIN_LOG_FILEPATH=${output_slice_dir}/inference.log \
    CHRONOSTRAIN_CACHE_DIR=${CHRONOSTRAIN_CACHE_DIR} \
    chronostrain advi \
    -r "${output_slice_dir}/filtered_input.csv" \
    -o ${output_slice_dir} \
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

  # ====== Record runtime
  end_time=$(date +%s%N)
  elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
  echo "${elapsed_time}" > $runtime_file

  t_idx=$(($t_idx+1))  # increment time index
done
