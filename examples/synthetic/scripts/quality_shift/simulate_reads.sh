#!/bin/bash
set -e
source settings.sh

seed=0

for (( q_shift = ${Q_SHIFT_MIN}; q_shift < ${Q_SHIFT_MAX}+1; q_shift += ${Q_SHIFT_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		seed=$((seed+1))

		trial_dir=$(get_trial_dir $q_shift $trial)
		read_dir=${trial_dir}/reads
		log_dir=${trial_dir}/logs

		echo "[Number of reads: ${N_READS}, qShift: ${q_shift}, trial #${trial}] -> ${trial_dir}"

		mkdir -p $log_dir
		mkdir -p $read_dir
		export CHRONOSTRAIN_LOG_FILEPATH="${log_dir}/read_sample.log"
		export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"

		python ${BASE_DIR}/helpers/sample_reads.py \
		--out_dir $read_dir \
		--abundance_path $GROUND_TRUTH \
		--num_reads $N_READS \
		--profiles $READ_PROFILE_PATH $READ_PROFILE_PATH \
		--read_len $READ_LEN \
		--seed ${seed} \
		--qShift ${q_shift} \
		--num_cores $N_CORES
	done
done