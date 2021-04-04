#!/bin/bash
set -e

#if [ -z ${PROJECT_DIR} ]; then
#	echo "Variable 'PROJECT_DIR' is not set. Exiting."
#	exit 1
#else
#	echo "PROJECT_DIR=${PROJECT_DIR}"
#fi

PROJECT_DIR="/PHShome/yk847/chronostrain"
CHRONOSTRAIN_DATA_DIR="/data/cctm/chronostrain"

# =====================================
# Command line args
NUM_READS=$1
TRIAL=$2
METHOD=$3
NUM_ITERS=$4


# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
TRUE_ABUNDANCE_PATH="${BASE_DIR}/files/true_abundances.csv"

CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/reads_${NUM_READS}_trial_${TRIAL}/chronostrain.log"
RUNS_DIR="${CHRONOSTRAIN_DATA_DIR}/runs"
READ_LEN=150
TRIAL_DIR="${RUNS_DIR}/trials/reads_${NUM_READS}_trial_${TRIAL}"

READS_DIR="${TRIAL_DIR}/simulated_reads"
OUTPUT_DIR="${TRIAL_DIR}/output/chronostrain"
OUTPUT_FILENAME="abundances.out"
SEED=$TRIAL
# =====================================

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH

# =========== Run chronostrain. ==================
python $PROJECT_DIR/scripts/run_inference.py \
--reads_dir $READS_DIR \
--true_abundance_path $TRUE_ABUNDANCE_PATH \
--method $METHOD \
--read_length $READ_LEN \
--seed $SEED \
-lr 0.001 \
--iters $NUM_ITERS \
--out_dir $OUTPUT_DIR \
--abundances_file $OUTPUT_FILENAME
# ================================================
