#!/bin/bash
set -e

source settings.sh

mkdir -p $CHRONOSTRAIN_LSF_DIR
mkdir -p $METAPHLAN_LSF_DIR
mkdir -p $CHRONOSTRAIN_LSF_OUTPUT_DIR
mkdir -p $METAPHLAN_LSF_OUTPUT_DIR

# ================ LSF creation ===========================
for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		# =============== Trial-specific settings ===================
		CHRONOSTRAIN_LSF_PATH="${CHRONOSTRAIN_LSF_DIR}/reads_${n_reads}_trial_${trial}-chronostrain.lsf"
		METAPHLAN_LSF_PATH="${METAPHLAN_LSF_DIR}/reads_${n_reads}_trial_${trial}-metaphlan.lsf"

    TRIAL_DIR="${RUNS_DIR}/reads_${n_reads}/trial_${trial}"
    READS_DIR="${TRIAL_DIR}/simulated_reads"
		CHRONOSTRAIN_OUTPUT_DIR="${TRIAL_DIR}/output/chronostrain"
		METAPHLAN_OUTPUT_DIR="${TRIAL_DIR}/output/metaphlan"
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
echo "Output dir: ${CHRONOSTRAIN_OUTPUT_DIR}"

export CHRONOSTRAIN_DATA_DIR=${CHRONOSTRAIN_DATA_DIR}
export BASE_DIR=${BASE_DIR}
export CHRONOSTRAIN_INI=$CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI=$CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH=${CHRONOSTRAIN_DATA_DIR}/logs/reads_${n_reads}/trial_${trial}/chronostrain.log

python $PROJECT_DIR/scripts/run_inference.py \
--reads_dir $READS_DIR \
--true_abundance_path $TRUE_ABUNDANCE_PATH \
--method $CHRONOSTRAIN_METHOD \
--read_length $READ_LEN \
--seed $SEED \
-lr $CHRONOSTRAIN_LR \
--iters $CHRONOSTRAIN_NUM_ITERS \
--out_dir $CHRONOSTRAIN_OUTPUT_DIR \
--abundances_file $CHRONOSTRAIN_OUTPUT_FILENAME
EOFDOC

		# ============ Metaphlan LSF ============
		echo "Creating ${METAPHLAN_LSF_PATH}"
		cat <<- EOFDOC > $METAPHLAN_LSF_PATH
#!/bin/bash
#BSUB -J bench_metaphlan
#BSUB -o ${METAPHLAN_LSF_OUTPUT_DIR}/%J-metaphlan_${n_reads}_${trial}.out
#BSUB -e ${METAPHLAN_LSF_OUTPUT_DIR}/%J-metaphlan_${n_reads}_${trial}.err
#BSUB -q $LSF_METAPHLAN_QUEUE
#BSUB -n ${LSF_METAPHLAN_N_CORES}
#BSUB -M ${LSF_METAPHLAN_MEM}
#BSUB -R rusage[mem=${LSF_METAPHLAN_MEM}]

source activate ${CONDA_ENV}
source ${SETTINGS_PATH}

export READS_DIR=$READS_DIR
export OUTPUT_DIR=$METAPHLAN_OUTPUT_DIR
export METAPHLAN_DB=${METAPHLAN_DB}
export METAPHLAN_DB_INDEX=${METAPHLAN_DB_INDEX}

bash ${BASE_DIR}/scripts/helpers/run_metaphlan.sh
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
