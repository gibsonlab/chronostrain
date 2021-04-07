#!/bin/bash
set -e

source settings.sh

mkdir -p $READGEN_LSF_DIR
mkdir -p $READGEN_LSF_OUTPUT_DIR

for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		LSF_PATH="${READGEN_LSF_DIR}/sample_reads_${n_reads}_trial_${trial}.lsf"
		LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/reads_${n_reads}/trial_${trial}/readgen.log"
    TRIAL_DIR="${RUNS_DIR}/reads_${n_reads}/trial_${trial}"
    READS_DIR="${TRIAL_DIR}/simulated_reads"
    mkdir -p $READS_DIR
    SEED=$trial

    echo "[Number of reads: ${n_reads}, trial #${trial}] -> ${LSF_PATH}"

		cat <<- EOFDOC > $LSF_PATH
#BSUB -J readgen
#BSUB -o ${READGEN_LSF_OUTPUT_DIR}/%J-readgen_${n_reads}_${trial}.out
#BSUB -e ${READGEN_LSF_OUTPUT_DIR}/%J-readgen_${n_reads}_${trial}.err
#BSUB -q ${LSF_READGEN_QUEUE}
#BSUB -n ${LSF_READGEN_N_CORES}
#BSUB -M ${LSF_READGEN_MEM}
#BSUB -R rusage[mem=${LSF_READGEN_MEM}]

export BASE_DIR=${BASE_DIR}
export CHRONOSTRAIN_DATA_DIR=${CHRONOSTRAIN_DATA_DIR}
export CHRONOSTRAIN_INI=${CHRONOSTRAIN_INI}
export CHRONOSTRAIN_LOG_INI=${CHRONOSTRAIN_LOG_INI}
export CHRONOSTRAIN_LOG_FILEPATH=${LOG_FILEPATH}
export CHRONOSTRAIN_DB_DIR=${CHRONOSTRAIN_DB_DIR}

source activate ${CONDA_ENV}
python ${PROJECT_DIR}/scripts/readgen.py \
--num_reads $n_reads \
--read_len $READ_LEN \
--out_dir $READS_DIR \
--profiles $READ_PROFILE_PATH $READ_PROFILE_PATH \
--abundance_path $TRUE_ABUNDANCE_PATH \
--seed $SEED \
--num_cores $LSF_READGEN_N_CORES
EOFDOC
  done
done

## ============== Submit all LSF jobs. ================
if [[ ${LSF_AUTO_SUBMIT} == 1 ]]
then
	for lsf_file in ${READGEN_LSF_DIR}/*.lsf
	do
		bsub < $lsf_file
	done
fi
