#!/bin/bash
set -e

source settings.sh

mkdir -p $CHRONOSTRAIN_LSF_DIR
mkdir -p $STRAINGE_LSF_DIR
mkdir -p $FILTER_LSF_DIR
mkdir -p $CHRONOSTRAIN_LSF_OUTPUT_DIR
mkdir -p $STRAINGE_LSF_OUTPUT_DIR
mkdir -p $FILTER_LSF_OUTPUT_DIR

# ================ LSF creation ===========================
SEED=0
for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( quality_shift = ${Q_SHIFT_MIN}; quality_shift < ${Q_SHIFT_MAX}+1; quality_shift += ${Q_SHIFT_STEP} ));
	do
		for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
		do
			# =============== Trial-specific settings ===================
			CHRONOSTRAIN_LSF_PATH="${CHRONOSTRAIN_LSF_DIR}/reads_${n_reads}_qs_${quality_shift}_trial_${trial}.lsf"
			FILTER_LSF_PATH="${FILTER_LSF_DIR}/reads_${n_reads}_qs_${quality_shift}_trial_${trial}.lsf"
			STRAINGE_LSF_PATH="${STRAINGE_LSF_DIR}/reads_${n_reads}_qs_${quality_shift}_trial_${trial}.lsf"

			TRIAL_DIR="${RUNS_DIR}/reads_${n_reads}/qs_${quality_shift}/trial_${trial}"
			READS_DIR="${TRIAL_DIR}/simulated_reads"
			CHRONOSTRAIN_OUTPUT_DIR="${TRIAL_DIR}/output/chronostrain"
			STRAINGE_OUTPUT_DIR="${TRIAL_DIR}/output/strainge"
			SEED=$((SEED+1))

			# ============ Filter LSF ===========
			echo "Creating ${FILTER_LSF_PATH}"
			export CHRONOSTRAIN_LOG_FILEPATH=${CHRONOSTRAIN_DATA_DIR}/logs/reads_${n_reads}/qs_${quality_shift}/trial_${trial}/filter.log

			echo "n_reads: ${n_reads}"
			echo "trial: ${trial}"
			echo "reads dir: ${READS_DIR}"

			echo "Filtering reads."
			python ${PROJECT_DIR}/scripts/filter_timeseries.py -r "${READS_DIR}" -o "${READS_DIR}/filtered"

			# ============ Chronostrain LSF ============
			# Generate LSF files via heredoc.
			echo "Creating ${CHRONOSTRAIN_LSF_PATH}"
			mkdir -p ${CHRONOSTRAIN_OUTPUT_DIR}

			export CHRONOSTRAIN_LOG_FILEPATH=${CHRONOSTRAIN_DATA_DIR}/logs/reads_${n_reads}/qs_${quality_shift}/trial_${trial}/chronostrain.log
			export CHRONOSTRAIN_OUTPUT_DIR=${CHRONOSTRAIN_OUTPUT_DIR}
			export SEED=${SEED}
			export READS_DIR=${READS_DIR}/filtered

			echo "n_reads: ${n_reads}"
			echo "trial: ${trial}"
			echo "reads dir: ${READS_DIR}/filtered"
			echo "Output dir: ${CHRONOSTRAIN_OUTPUT_DIR}"

			bash run_chronostrain.sh

			# ============ StrainGE LSF -- Markers (filtered) ============
			echo "Creating ${STRAINGE_LSF_PATH}"

			echo "n_reads: ${n_reads}"
			echo "trial: ${trial}"

			echo "================ Markers Only (Filtered) ===================="
			export READS_DIR=${READS_DIR}/filtered
			export STRAINGE_DB_PATH=${STRAINGE_MARKERS_DB_PATH}
			export STRAINGE_OUTPUT_DIR="${STRAINGE_OUTPUT_DIR}/markers_filtered"
			export EXTENSION="fq"
			bash run_strainge.sh

			echo "================ Markers Only (Unfiltered) ===================="
			export READS_DIR=${READS_DIR}
			export STRAINGE_DB_PATH=${STRAINGE_MARKERS_DB_PATH}
			export STRAINGE_OUTPUT_DIR="${STRAINGE_OUTPUT_DIR}/markers_unfiltered"
			export EXTENSION="fastq"
			bash run_strainge.sh

			echo "================ Entire Ecoli genome ===================="
			export READS_DIR=${READS_DIR}
			export STRAINGE_DB_PATH=${STRAINGE_GENOMES_DB_PATH}
			export STRAINGE_OUTPUT_DIR="${STRAINGE_OUTPUT_DIR}/genomes"
			export EXTENSION="fastq"
			bash run_strainge.sh

			echo "================ Full DB ===================="
			export READS_DIR=${READS_DIR}
			export STRAINGE_DB_PATH=${STRAINGE_FULL_DB_PATH}
			export STRAINGE_OUTPUT_DIR="${STRAINGE_OUTPUT_DIR}/full"
			export EXTENSION="fastq"
			bash run_strainge.sh
		done
	done
done
