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
log_dir=${trial_dir}/logs

mkdir -p $log_dir
export CHRONOSTRAIN_LOG_FILEPATH="${log_dir}/chronostrain.log"
export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"

echo "[*] Running Chronostrain inference for n_reads: ${n_reads}, trial: ${trial}"
python $PROJECT_DIR/scripts/run_bbvi.py \
--reads_input "${read_dir}/filtered/filtered_input_files.csv" \
--out_dir ${output_dir} \
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
