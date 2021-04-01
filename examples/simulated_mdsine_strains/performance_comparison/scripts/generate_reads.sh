#!/bin/bash
set -e

if [ -z ${PROJECT_DIR} ]; then
	echo "Variable 'PROJECT_DIR' is not set. Exiting."
	exit 1
else
	echo "PROJECT_DIR=${PROJECT_DIR}"
fi

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

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH

for n_reads in {0..10..2}
do
  for trial in {1..N_TRIALS}
  do
    echo "[Number of reads: ${n_reads}, trial #${trial}]"

    TRIAL_DIR="${RUNS_DIR}/trials/reads_${n_reads}_trial_${trial}"
    READS_DIR="${TRIAL_DIR}/simulated_reads"
    mkdir -p $READS_DIR
    SEED=$trial

    # ================== Generate the reads. ================
    # TODO use Zack's sampler.
    python $PROJECT_DIR/scripts/simulate_reads.py \
    --seed $SEED \
    --out_dir $READS_DIR \
    --abundance_path $TRUE_ABUNDANCE_PATH \
    --num_reads $n_reads \
    --read_length $READ_LEN
    # =======================================================
  done
done
