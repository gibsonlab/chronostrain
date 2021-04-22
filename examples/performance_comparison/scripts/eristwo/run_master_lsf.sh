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
			SEED=$trial

			# ============ Filter LSF ===========
			echo "Creating ${FILTER_LSF_PATH}"
			cat <<- EOFDOC > $FILTER_LSF_PATH
#!/bin/bash
#BSUB -J filter
#BSUB -o ${FILTER_LSF_OUTPUT_DIR}/filter_reads_${n_reads}_qs_${quality_shift}_trial_${trial}-%J.out
#BSUB -e ${FILTER_LSF_OUTPUT_DIR}/filter_reads_${n_reads}_qs_${quality_shift}_trial_${trial}-%J.err
#BSUB -q ${LSF_FILTER_QUEUE}
#BSUB -n ${LSF_FILTER_N_CORES}
#BSUB -M ${LSF_FILTER_MEM}
#BSUB -R rusage[mem=${LSF_FILTER_MEM}]

source activate ${CONDA_ENV}
export CHRONOSTRAIN_LOG_FILEPATH=${CHRONOSTRAIN_DATA_DIR}/logs/reads_${n_reads}/qs_${quality_shift}/trial_${trial}/filter.log

echo "n_reads: ${n_reads}"
echo "trial: ${trial}"
echo "reads dir: ${READS_DIR}"

source ${BASE_DIR}/scripts/eristwo/settings.sh

echo "Filtering reads."
python ${PROJECT_DIR}/scripts/filter.py -r "${READS_DIR}" -o "${READS_DIR}/filtered"
EOFDOC

			# ============ Chronostrain LSF ============
			# Generate LSF files via heredoc.
			echo "Creating ${CHRONOSTRAIN_LSF_PATH}"
			cat <<- EOFDOC > $CHRONOSTRAIN_LSF_PATH
#!/bin/bash
#BSUB -J bench_chronostrain
#BSUB -o ${CHRONOSTRAIN_LSF_OUTPUT_DIR}/chronostrain_reads_${n_reads}_qs_${quality_shift}_trial_${trial}-%J.out
#BSUB -e ${CHRONOSTRAIN_LSF_OUTPUT_DIR}/chronostrain_reads_${n_reads}_qs_${quality_shift}_trial_${trial}-%J.err
#BSUB -q $LSF_CHRONOSTRAIN_QUEUE
#BSUB -n ${LSF_CHRONOSTRAIN_N_CORES}
#BSUB -M ${LSF_CHRONOSTRAIN_MEM}
#BSUB -R rusage[mem=${LSF_CHRONOSTRAIN_MEM}]

source activate ${CONDA_ENV}
mkdir -p ${CHRONOSTRAIN_OUTPUT_DIR}

export CHRONOSTRAIN_LOG_FILEPATH=${CHRONOSTRAIN_DATA_DIR}/logs/reads_${n_reads}/qs_${quality_shift}/trial_${trial}/chronostrain.log
export CHRONOSTRAIN_OUTPUT_DIR=${CHRONOSTRAIN_OUTPUT_DIR}
export SEED=${SEED}
export READS_DIR=${READS_DIR}

echo "n_reads: ${n_reads}"
echo "trial: ${trial}"
echo "reads dir: ${READS_DIR}"
echo "Output dir: ${CHRONOSTRAIN_OUTPUT_DIR}"

cd ${BASE_DIR}/scripts/eristwo
bash run_chronostrain.sh
EOFDOC

			# ============ StrainGE LSF ============
			echo "Creating ${STRAINGE_LSF_PATH}"
			cat <<- EOFDOC > ${STRAINGE_LSF_PATH}
#!/bin/bash
#BSUB -J bench_strainGE
#BSUB -o ${STRAINGE_LSF_OUTPUT_DIR}/strainge_reads_${n_reads}_qs_${quality_shift}_trial_${trial}-%J.out
#BSUB -e ${STRAINGE_LSF_OUTPUT_DIR}/strainge_reads_${n_reads}_qs_${quality_shift}_trial_${trial}-%J.err
#BSUB -q ${LSF_STRAINGE_QUEUE}
#BSUB -n ${LSF_STRAINGE_N_CORES}
#BSUB -M ${LSF_STRAINGE_MEM}
#BSUB -R rusage[mem=${LSF_STRAINGE_MEM}]

source activate ${CONDA_ENV}
mkdir -p ${STRAINGE_OUTPUT_DIR}

echo "n_reads: ${n_reads}"
echo "trial: ${trial}"
echo "reads dir: ${READS_DIR}"
echo "Output dir: ${STRAINGE_OUTPUT_DIR}"

export READS_DIR=${READS_DIR}
export STRAINGE_OUTPUT_DIR="${STRAINGE_OUTPUT_DIR}"
export CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/reads_${n_reads}/qs_${quality_shift}/trial_${trial}/strainge_plot.log"

cd ${BASE_DIR}/scripts/eristwo
bash run_strainge.sh
EOFDOC
		done
	done
done


## ============== Submit all LSF jobs. ================
if [[ ${LSF_AUTO_SUBMIT} == 1 ]]
then
	echo "Submitting Chronostrain LSF jobs."
	for lsf_file in ${CHRONOSTRAIN_LSF_DIR}/*.lsf
	do
		bsub < $lsf_file
	done

	echo "Submitting MetaPhlAn LSF jobs."
	for lsf_file in ${METAPHLAN_LSF_DIR}/*.lsf
	do
		bsub < $lsf_file
	done
fi
