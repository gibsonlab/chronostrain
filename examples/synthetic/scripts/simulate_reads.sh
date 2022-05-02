#!/bin/bash
set -e
source settings.sh

seed=0

for n_reads in 10000 50000 100000 500000 1000000
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		seed=$((seed+1))

		trial_dir=$(get_trial_dir $n_reads $trial)
		read_dir=${trial_dir}/reads
		log_dir=${trial_dir}/logs

		echo "[Number of reads: ${n_reads}, qShift: ${quality_shift}, trial #${trial}] -> ${trial_dir}"

		mkdir -p $log_dir
		mkdir -p $read_dir
		export CHRONOSTRAIN_LOG_FILEPATH="${log_dir}/read_sample.log"

		python ${BASE_DIR}/helpers/sample_reads.py \
		--out_dir $read_dir \
		--abundance_path $GROUND_TRUTH \
		--num_reads $n_reads \
		--profiles $READ_PROFILE_PATH $READ_PROFILE_PATH \
		--read_len $READ_LEN \
		--seed ${seed} \
		--num_cores $N_CORES
	done
done