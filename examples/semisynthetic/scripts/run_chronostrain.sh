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
runtime_file=${trial_dir}/output/chronostrain_runtime.txt

if [[ -f $runtime_file ]]; then
	echo "[*] Skipping Chronostrain Inference (n_reads: ${n_reads}, trial: ${trial})"
	exit 0
fi

mkdir -p $output_dir
export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"
export CHRONOSTRAIN_LOG_FILEPATH="${output_dir}/chronostrain.log"

echo "[*] Running Chronostrain inference for n_reads: ${n_reads}, trial: ${trial}"
start_time=$(date +%s%N)  # nanoseconds

python $PROJECT_DIR/scripts/run_advi.py \
--reads_input "${output_dir}/filtered/filtered_input_files.csv" \
--out_dir ${output_dir} \
--correlation_mode "full" \
--seed ${INFERENCE_SEED} \
--iters $CHRONOSTRAIN_NUM_ITERS \
--epochs $CHRONOSTRAIN_NUM_EPOCHS \
--decay_lr $CHRONOSTRAIN_DECAY_LR \
--lr_patience ${CHRONOSTRAIN_LR_PATIENCE} \
--min_lr ${CHRONOSTRAIN_MIN_LR} \
--learning_rate $CHRONOSTRAIN_LR \
--num_samples $CHRONOSTRAIN_NUM_SAMPLES \
--read_batch_size $CHRONOSTRAIN_READ_BATCH_SZ \
--plot_format "pdf" \
--plot_elbo


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file
