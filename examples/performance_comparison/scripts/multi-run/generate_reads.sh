#!/bin/bash
set -e

source settings.sh

mkdir -p $READGEN_LSF_DIR
mkdir -p $READGEN_LSF_OUTPUT_DIR
SEED=0

for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( quality_shift = ${Q_SHIFT_MIN}; quality_shift < ${Q_SHIFT_MAX}+1; quality_shift += ${Q_SHIFT_STEP} ));
	do
		for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
		do
			LSF_PATH="${READGEN_LSF_DIR}/sample_reads_${n_reads}_qs_${quality_shift}_trial_${trial}.lsf"
			LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/reads_${n_reads}/qs_${quality_shift}/trial_${trial}/readgen.log"
			TRIAL_DIR="${RUNS_DIR}/reads_${n_reads}/qs_${quality_shift}/trial_${trial}"
			READS_DIR="${TRIAL_DIR}/simulated_reads"
			SEED=$((SEED+1))

			echo "[Number of reads: ${n_reads}, qShift: ${quality_shift}, trial #${trial}] -> ${LSF_PATH}"

			export BASE_DIR=${BASE_DIR}
			export CHRONOSTRAIN_DATA_DIR=${CHRONOSTRAIN_DATA_DIR}
			export CHRONOSTRAIN_INI=${CHRONOSTRAIN_INI}
			export CHRONOSTRAIN_LOG_INI=${CHRONOSTRAIN_LOG_INI}
			export CHRONOSTRAIN_LOG_FILEPATH=${LOG_FILEPATH}
			export CHRONOSTRAIN_DB_DIR=${CHRONOSTRAIN_DB_DIR}

			echo "Output dir: ${READS_DIR}"
			echo "qShift: ${quality_shift}"
			echo "seed: ${SEED}"

			mkdir -p $READS_DIR

			python ${PROJECT_DIR}/scripts/readgen.py \
			--num_reads $n_reads \
			--read_len $READ_LEN \
			--out_dir $READS_DIR \
			--profiles $READ_PROFILE_PATH $READ_PROFILE_PATH \
			--abundance_path $TRUE_ABUNDANCE_PATH \
			--seed $SEED \
			--qShift ${quality_shift} \
			--num_cores $LSF_READGEN_N_CORES
  	done
  done
done
