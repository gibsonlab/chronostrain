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
filter_file=${trial_dir}/output/chronostrain_filter_runtime.txt

if [ -f $runtime_file ]; then
	echo "[*] Skipping Chronostrain Inference (n_reads: ${n_reads}, trial: ${trial})"
	exit 0
fi

if [ ! -f $filter_file ]; then
	echo "[*] Filtered result not found."
	exit 1
fi

mkdir -p $output_dir
export CHRONOSTRAIN_CACHE_DIR="${output_dir}/cache"
export CHRONOSTRAIN_LOG_FILEPATH="${output_dir}/chronostrain.log"

if [ -d $CHRONOSTRAIN_CACHE_DIR ]; then
	echo "[*] Clearing cache."
	rm -rf $CHRONOSTRAIN_CACHE_DIR
fi

echo "[*] Using database ${CHRONOSTRAIN_DB_JSON}"
echo "[*] Running Chronostrain inference for n_reads: ${n_reads}, trial: ${trial}"
start_time=$(date +%s%N)  # nanoseconds

chronostrain advi \
	-r "${output_dir}/filtered/filtered_input_files.csv" \
	-o ${output_dir} \
	--correlation-mode "full" \
	--iters $CHRONOSTRAIN_NUM_ITERS \
	--epochs $CHRONOSTRAIN_NUM_EPOCHS \
	--decay-lr $CHRONOSTRAIN_DECAY_LR \
	--lr-patience ${CHRONOSTRAIN_LR_PATIENCE} \
	--loss-tol ${CHRONOSTRAIN_LOSS_TOL} \
	--learning-rate $CHRONOSTRAIN_LR \
	--num-samples $CHRONOSTRAIN_NUM_SAMPLES \
	--read-batch-size $CHRONOSTRAIN_READ_BATCH_SZ \
	--min-lr ${CHRONOSTRAIN_MIN_LR} \
	--plot-format "pdf" \
	--plot-elbo \
	--prune-strains \
	--with-zeros \
	--accumulate-gradients


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file
