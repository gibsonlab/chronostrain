#!/bin/bash
set -e

if [ -z ${PROJECT_DIR} ]; then
	echo "Variable 'PROJECT_DIR' is not set. Exiting."
	exit 1
else
	echo "PROJECT_DIR=${PROJECT_DIR}"
fi

# ======================================

N_READS_MIN=$1
N_READS_MAX=$2
N_READS_STEP=$3
N_TRIALS=$4

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

CHRONOSTRAIN_INI="${BASE_DIR}/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${BASE_DIR}/logs/read_sample.log"

TRUE_ABUNDANCE_PATH="${BASE_DIR}/true_abundances.csv"

RUNS_DIR="${BASE_DIR}/runs"
READ_LEN=150
# =====================================
# The location of the ReadGenAndFiltering repo for sampling
READGEN_DIR="/PHShome/yk847/Read-Generation-and-Filtering"
# =====================================

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH

for (( n_reads = ${N_READS_MIN}; n_reads < ${N_READS_MAX}+1; n_reads += ${N_READS_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
    echo "[Number of reads: ${n_reads}, trial #${trial}]"

    TRIAL_DIR="${RUNS_DIR}/trials/reads_${n_reads}_trial_${trial}"
    READS_DIR="${TRIAL_DIR}/simulated_reads"
    mkdir -p $READS_DIR
    SEED=$trial

    # ================== Generate the reads. ================
		python ${PROJECT_DIR}/scripts/readgen.py \
		$n_reads \
		$READ_LEN \
		$trial \
		$READGEN_DIR/profiles/HiSeqReference \
		$READGEN_DIR/profiles/HiSeqReference \
		$TRUE_ABUNDANCE_PATH \
		$READS_DIR \
		$SEED
    # =======================================================
  done
done
