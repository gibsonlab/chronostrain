#!/bin/bash
set -e
source settings.sh

# ============ Requires arguments:
q_shift=$1
trial=$2

if [ -z "$q_shift" ]
then
	echo "var \"q_shift\" is empty"
	exit 1
fi

if [ -z "$trial" ]
then
	echo "var \"trial\" is empty"
	exit 1
fi

# ============ script body:
trial_dir=$(get_trial_dir $q_shift $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/chronostrain
log_dir=${trial_dir}/logs

mkdir -p $log_dir
export CHRONOSTRAIN_LOG_FILEPATH="${log_dir}/chronostrain.log"
export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"

echo "[*] Running Chronostrain inference for q_shift: ${q_shift}, trial: ${trial}"
chronostrain advi \
	-r "${read_dir}/filtered/filtered_input_files.csv" \
	-o ${output_dir} \
	--seed ${INFERENCE_SEED} \
	--iters $CHRONOSTRAIN_NUM_ITERS \
	--epochs $CHRONOSTRAIN_NUM_EPOCHS \
	--decay-lr $CHRONOSTRAIN_DECAY_LR \
	--lr-patience 10 \
	--min-lr 1e-5 \
	--learning-rate $CHRONOSTRAIN_LR \
	--num-samples $CHRONOSTRAIN_NUM_SAMPLES \
	--read-batch_size $CHRONOSTRAIN_READ_BATCH_SZ \
	--plot-format "pdf" \
	--plot-elbo
