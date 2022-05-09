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
log_dir=${trial_dir}/logs

mkdir -p $log_dir
mkdir -p $read_dir
export CHRONOSTRAIN_LOG_FILEPATH="${log_dir}/filter.log"
export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"

echo "[*] Filtering reads for n_reads: ${n_reads}, trial: ${trial}"
python $PROJECT_DIR/scripts/filter_timeseries.py \
--reads_input "${read_dir}/input_files.csv" \
-o "${read_dir}/filtered/" \
--frac_identity_threshold 0.975 \
--error_threshold 1.0 \
--num_threads 4
