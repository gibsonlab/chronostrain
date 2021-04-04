#!/bin/bash
set -e

PROJECT_DIR="/PHShome/yk847/chronostrain"
CHRONOSTRAIN_DATA_DIR="/data/cctm/chronostrain"

# ======================================
N_READS_MIN=1000000
N_READS_MAX=10000000
N_READS_STEP=1000000
N_TRIALS=10

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"

TRUE_ABUNDANCE_PATH="${BASE_DIR}/files/true_abundances.csv"

RUNS_DIR="${CHRONOSTRAIN_DATA_DIR}/runs"
READ_LEN=150
READ_PROFILE_PATH="${BASE_DIR}/files/HiSeqReference"

LSF_QUEUE="normal"
CONDA_ENV="chronostrain"
LSF_MEM=10000
LSF_N_CORES=1
LSF_DIR="${CHRONOSTRAIN_DATA_DIR}/lsf_files/readgen"
LSF_OUTPUT_DIR="${LSF_DIR}/output"
# =====================================

mkdir -p $LSF_DIR
mkdir -p $LSF_OUTPUT_DIR

for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
    echo "[Number of reads: ${n_reads}, trial #${trial}]"
		LSF_PATH="${LSF_DIR}/sample_reads_${n_reads}_trial_${trial}.lsf"
		LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/readgen/reads_${n_reads}_trial_${trial}.log"

    TRIAL_DIR="${RUNS_DIR}/trials/reads_${n_reads}_trial_${trial}"
    READS_DIR="${TRIAL_DIR}/simulated_reads"
    mkdir -p $READS_DIR
    SEED=$trial

		cat <<- EOFDOC > $LSF_PATH
#BSUB -J readgen
#BSUB -o ${LSF_OUTPUT_DIR}/%J-readgen_${n_reads}_${trial}.out
#BSUB -e ${LSF_OUTPUT_DIR}/%J-readgen_${n_reads}_${trial}.err
#BSUB -q ${LSF_QUEUE}
#BSUB -n ${LSF_N_CORES}
#BSUB -M ${LSF_MEM}
#BSUB -R rusage[mem=${LSF_MEM}]

export BASE_DIR=${BASE_DIR}
export CHRONOSTRAIN_INI=${CHRONOSTRAIN_INI}
export CHRONOSTRAIN_LOG_INI=${CHRONOSTRAIN_LOG_INI}
export CHRONOSTRAIN_LOG_FILEPATH=${LOG_FILEPATH}

python ${PROJECT_DIR}/scripts/readgen.py \
--num_reads $n_reads \
--read_len $READ_LEN \
--out_dir $READS_DIR \
--profiles $READ_PROFILE_PATH $READ_PROFILE_PATH \
--abundance_path $TRUE_ABUNDANCE_PATH \
--seed $SEED \
--num_cores $LSF_N_CORES
EOFDOC
  done
done
