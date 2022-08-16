#!/bin/bash
set -e
source settings.sh

# ============ Requires arguments:
q_shift=$1
trial=$2

if [ -z "q_shift" ]
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
log_dir=${trial_dir}/logs

mkdir -p $log_dir
mkdir -p $read_dir
export CHRONOSTRAIN_LOG_FILEPATH="${log_dir}/filter.log"
export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"

echo "[*] Filtering reads for q_shift: ${q_shift}, trial: ${trial}"
chronostrain filter \
-r "${read_dir}/input_files.csv" \
-o "${read_dir}/filtered"
