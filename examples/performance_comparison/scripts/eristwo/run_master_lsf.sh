#!/bin/bash
set -e

source settings.sh

mkdir -p $CHRONOSTRAIN_LSF_DIR
mkdir -p $STRAINGE_LSF_DIR
mkdir -p $CHRONOSTRAIN_LSF_OUTPUT_DIR
mkdir -p $STRAINGE_LSF_OUTPUT_DIR

# ================ LSF creation ===========================
for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		# =============== Trial-specific settings ===================
		CHRONOSTRAIN_LSF_PATH="${CHRONOSTRAIN_LSF_DIR}/reads_${n_reads}_trial_${trial}-chronostrain.lsf"
		STRAINGE_LSF_PATH="${STRAINGE_LSF_DIR}/reads_${n_reads}_trial_${trial}-strainge.lsf"

    TRIAL_DIR="${RUNS_DIR}/reads_${n_reads}/trial_${trial}"
    READS_DIR="${TRIAL_DIR}/simulated_reads"
		CHRONOSTRAIN_OUTPUT_DIR="${TRIAL_DIR}/output/chronostrain"
		STRAINGE_OUTPUT_DIR="${TRIAL_DIR}/output/strainge"
		SEED=$trial

		# ============ Chronostrain LSF ============
		# Generate LSF files via heredoc.
		echo "Creating ${CHRONOSTRAIN_LSF_PATH}"
		cat <<- EOFDOC > $CHRONOSTRAIN_LSF_PATH
#!/bin/bash
#BSUB -J bench_chronostrain
#BSUB -o ${CHRONOSTRAIN_LSF_OUTPUT_DIR}/%J-chronostrain_${n_reads}_${trial}-%J.out
#BSUB -e ${CHRONOSTRAIN_LSF_OUTPUT_DIR}/%J-chronostrain_${n_reads}_${trial}-%J.err
#BSUB -q $LSF_CHRONOSTRAIN_QUEUE
#BSUB -n ${LSF_CHRONOSTRAIN_N_CORES}
#BSUB -M ${LSF_CHRONOSTRAIN_MEM}
#BSUB -R rusage[mem=${LSF_CHRONOSTRAIN_MEM}]

source activate ${CONDA_ENV}
mkdir -p $CHRONOSTRAIN_OUTPUT_DIR

echo "n_reads: ${n_reads}"
echo "trial: ${trial}"
echo "reads dir: ${READS_DIR}"
echo "Output dir: ${CHRONOSTRAIN_OUTPUT_DIR}"

export CHRONOSTRAIN_DATA_DIR=${CHRONOSTRAIN_DATA_DIR}
export BASE_DIR=${BASE_DIR}
export CHRONOSTRAIN_INI=$CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI=$CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH=${CHRONOSTRAIN_DATA_DIR}/logs/reads_${n_reads}/trial_${trial}/chronostrain.log

echo "Filtering reads."
python ${PROJECT_DIR}/scripts/filter.py \
-r "${READS_DIR}" \
-o "${READS_DIR}/filtered"

echo "Running inference."
python $PROJECT_DIR/scripts/run_inference.py \
--reads_dir ${READS_DIR}/filtered \
--true_abundance_path $TRUE_ABUNDANCE_PATH \
--method $CHRONOSTRAIN_METHOD \
--read_length $READ_LEN \
--seed $SEED \
-lr $CHRONOSTRAIN_LR \
--iters $CHRONOSTRAIN_NUM_ITERS \
--out_dir $CHRONOSTRAIN_OUTPUT_DIR \
--abundances_file $CHRONOSTRAIN_OUTPUT_FILENAME \
--num_cores
EOFDOC

		# ============ StrainGE LSF ============
		echo "Creating ${STRAINGE_LSF_PATH}"
		cat <<- EOFDOC > ${STRAINGE_LSF_PATH}
#!/bin/bash
#BSUB -J bench_strainGE
#BSUB -o ${STRAINGE_LSF_OUTPUT_DIR}/%J-chronostrain_${n_reads}_${trial}-%J.out
#BSUB -e ${STRAINGE_LSF_OUTPUT_DIR}/%J-chronostrain_${n_reads}_${trial}-%J.err
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

bash ${BASE_DIR}/scripts/eristwo/run_strainge.sh
EOFDOC

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
